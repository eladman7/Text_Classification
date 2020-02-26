# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:14:50 2017

@author: B
"""
import sys
sys.path.append('/root/maneltnt/Models/')
import numpy as np
np.random.seed(123)
import argparse
import Models , PageLoadBatches
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import glob
import cv2
import os

def find_word_in_path(path, word, separator):
    splitted_filename = path.split(separator)
    i=0
    index_of_patch = -1
    for split in splitted_filename:
        if word in split:
            index_of_patch = i
        i = i + 1
    return index_of_patch
    
def combine_splitted_arr_upto_ind(arr, index_to_remove_from, combine_char):
    lst_wo_suffix = map(lambda x: x.split(combine_char)[0:index_to_remove_from] ,arr)
    lst = list(map(lambda x: combine_char.join(x), lst_wo_suffix))
    return lst
    

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str,default ="bestweights"  )
parser.add_argument("--train_images", type = str, default ="hdata/ptrain/"  )
parser.add_argument("--train_annotations", type = str, default = "hdata/pltrain/"  )
parser.add_argument("--n_classes", type=int, default = 2 )
parser.add_argument("--input_height", type=int , default = 320  )
parser.add_argument("--input_width", type=int , default = 320 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "ldata/pvalidation/")
parser.add_argument("--val_annotations", type = str , default = "ldata/plvalidation/")

parser.add_argument("--test_images", type = str , default = "ldata/ptest/")
#parser.add_argument("--test_annotations", type = str , default = "ldata/pltest/")
parser.add_argument("--output_path", type = str , default = "pprediction/")



parser.add_argument("--epochs", type = int, default = 50 )
parser.add_argument("--batch_size", type = int, default = 16 )
parser.add_argument("--val_batch_size", type = int, default = 16 )
parser.add_argument("--test_batch_size", type = int, default = 16 )

#parser.add_argument("--load_weights", type = str , default = 'asarpagesegprebestweigts')

parser.add_argument("--model_name", type = str , default = "fcn8")
parser.add_argument("--optimizer_name", type = str , default = "sgd")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
#load_weights = args.load_weights
test_images_path = args.test_images
#test_segs_path = args.test_annotations
test_batch_size = args.test_batch_size

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = {'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
sgd = optimizers.SGD(lr=0.001)
#adm=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

m.compile(loss='categorical_crossentropy',
      optimizer= sgd,
      metrics=['accuracy'])

#
#if len( load_weights ) > 0:
#    print("loading initial weights")
#    m.load_weights(load_weights)


print ( m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

G  = PageLoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


if validate:
	G2  = PageLoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

mcp=ModelCheckpoint( filepath=save_weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True,verbose=1)
evaluate=0
if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 1  , epochs=1 )
else:
    for ep in range( epochs ):
        m.fit_generator( G , 500 , validation_data=G2 , validation_steps=5,  epochs=50,callbacks=[mcp] )
        evaluate=evaluate+1
        if evaluate%1==0:
            print('loading test images')
            images = sorted(glob.glob( test_images_path + "/*.jpg"  ) + glob.glob( test_images_path+ "/*.png"  ) +  glob.glob( test_images_path + "/*.jpeg"  ))
            images.sort()
    
            colors=[(255,255,255),(0,0,255),(255,0,0)]
    
            for imgName in images:
                outName = imgName.replace( test_images_path ,  args.output_path )
                X = PageLoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )
                pr = m.predict( np.array([X]) )[0]
                pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
                seg_img = np.zeros( ( output_height , output_width , 3  ) )
                for c in range(n_classes):
                    seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
                    #seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
                    seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
                seg_img = cv2.resize(seg_img  , (input_width , input_height ))
                cv2.imwrite(  outName , seg_img )
            
            print('ordering the predictions')
            folder='pprediction/'
            ofolder='opprediction/'
            for p in sorted(os.listdir(folder)):
                # assuming patch files names looks like: [prefix...][_patch]                        
                #original: pn=p.split('_')[3][:-4].replace('patch','')
                index_of_patch = find_word_in_path(p, 'patch', '_')
                pn=p.split('_')[index_of_patch][:-4].replace('patch','')
                l=len(pn)
                add= 4-l
                npa=''
                if add==1:
                    npa=p.replace('patch'+pn,'patch0'+pn)
                if add==2:
                    npa=p.replace('patch'+pn,'patch00'+pn)
                if add==3:
                    npa=p.replace('patch'+pn,'patch000'+pn)
           
                os.rename(folder+p,ofolder+npa)
            
            print('combining the predictions')
            patchSize=320
            patchNumber=0
            predictions='opprediction/'
            original='ldata/ltest/'
    
            paths = sorted(glob.glob(predictions+ "*.png" ))
            # original line: pages=[item.split('_')[0]+'_'+item.split('_')[1] for item in paths]
            pages= combine_splitted_arr_upto_ind(paths, index_of_patch, '_')
            oldpage=pages[0]
            g=[[]]
            i=0
            pathc=0
            for page in pages:
                if page==oldpage:
                    g[i].append(paths[pathc])
                    pathc=pathc+1
                else:
                    i=i+1
                    g.append([])
                    g[i].append(paths[pathc])
                    pathc=pathc+1
                    oldpage=page
    
    
            c2=0      
            for group in g:
                print('group[0]',group[0])
                oi=group[0].split('/')[1][:-14]+'.png'
                print('oi',oi)
                originalPage=cv2.imread(original+oi,0)
                rows,cols=originalPage.shape
                x=rows//patchSize
                y=cols//patchSize
                sx=x*patchSize
                sy=y*patchSize
                ni=np.zeros((int(sx),int(sy),3))+255
                for i in range(0,sx,patchSize):
                    for j in range(0,sy,patchSize):
                        ni[i:i+patchSize,j:j+patchSize]=cv2.imread(paths[c2],1)
                        c2=c2+1
                cv2.imwrite('out/'+group[0].split('/')[1][:-14]+'.png',ni)
            
            print('calculating the fmeasures')
            truefolder='ldata/ltest/'
            predfolder='out/'
            epsilon=0.0000000001
            fmain=[]
            #fside=[]
            for p in os.listdir(truefolder):
                print(p)
                true=cv2.imread(truefolder+p,0)
                pred=cv2.imread(predfolder+p[:-4]+'.png',1)
                
                rows,cols,ch=pred.shape
                print(pred.shape)
            
                for i in range(rows):
                    for j in range(cols):
                         # 3rd channel is always zero in our case
                         # original: if pred[i,j,0]==255 and pred[i,j,1]==255 and pred[i,j,2]==255:
                        if pred[i,j,0]==255 and pred[i,j,1]==255:
                            pass
                         # original: if pred[i,j,0]==255 and pred[i,j,1]<255 and pred[i,j,2]<255:
                        if pred[i,j,0]==255 and pred[i,j,1]<255:
                            pred[i,j,0]=0
                            pred[i,j,1]=0
#                            # pred[i,j,2]=0
                         # original: if pred[i,j,0]<255 and pred[i,j,1]<255 and pred[i,j,2]==255: 
                        if pred[i,j,0]<255 and pred[i,j,1]<255:
                            pred[i,j,0]=128
                            pred[i,j,1]=128
#                            # pred[i,j,2]=128
                pred[pred<128]=0
                p1=pred>128
                p2=pred<255
                pred[p1*p2]=128
            
                pred=pred[:,:,0]
            
                rows,cols=pred.shape
                print(pred.shape)
        
                mallp=0
                mtp=0
                mfp=0
                mfn=0
                #sallp=0
                #stp=0
                #sfp=0
                #sfn=0
                for i in range(rows):
                    for j in range(cols):
                        if true[i,j]==0:
                            mallp=mallp+1
                        # original line: if true[i,j]==0 and pred[i,j]==0:
                        if true[i,j]==0 and pred[i,j]==128:
                            mtp=mtp+1
                        # original line: if true[i,j]==128 and pred[i,j]==0:
                        if true[i,j]==255 and pred[i,j]==128:
                            mfp=mfp+1
                        if true[i,j]==0 and pred[i,j]==255:
                            mfn=mfn+1
                        #if true[i,j]==128:
                        #    sallp=sallp+1
                        #if true[i,j]==128 and pred[i,j]==128:
                        #    stp=stp+1
                        #if true[i,j]==0 and pred[i,j]==128:
                        #    sfp=sfp+1
                        #if true[i,j]==128 and pred[i,j]==0:
                        #    sfn=sfn+1
                if mallp>0:
                    fm=(2.*mtp+epsilon)/(2.*mtp+mfp+mfn+epsilon)
                    fmain.append(fm)
                    print(p)
                    print(fm)
                #if sallp>0:
                #    fs=(2.*stp+epsilon)/(2.*stp+sfp+sfn+epsilon)
                #    print(p)
                #    print(fs)
                #    fside.append(fs)
                print(p+' is finished')
                print('')
            print('main f measure:')
            print(np.mean(fmain))
            #print('side f measure:')
            #print(np.mean(fside))
 
     



        



        
    
