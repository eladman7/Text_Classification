# Text_Classification
About the code:
the following project is intended to learn and classify images into 2 classes: text and none text.
users will have to prepare labeled images i.e. black & white images where text areas are black and all other are white.
user will have to choose some of the raw and labeled images to serve as training data and some to test the alg with.

Project Structure:
hdata - holds the training raw and labeled data.
  ltrain - holds the labeled train images
  pltrain - holds the labeled patches of the train images
  train - holds raw train images
  ptrain - holds patches of raw train images
  HighDenseTrainPatchMaker.py - script to create train patches. uses train and ltrain contents and outputs relevant patches into ptrain                                   and pltrain correspondingly
ldata - holds the testing raw and labeled data.
  ltest - holds the labeled test images
  test - holds the raw test images
  ptest - holds the patches of raw test images
  pvalidation - holds validation patches
  plvalidation - holds labeled validation patches
  TestPatchMaker.py - script to create test patches use the contents of test and outputs relevant patches into ptest
  ValidationPatchMaker.py - script to create validation patches. uses test and ltest and outputs relevant patches into pvalidation and                                 plvalidation correspondingly
Models - contain the machine learning algorithm code
opprediction - used by the alg to store patches in order
pprediction - used by the alg to store prediction for each patch, none ordered
out - the alg prediction images
PageLoadBatches.py - script with aux functions
pagepredict.py - prediction aux functions
pagetrainf8.py - main script to run

Usage & Run
  Prepare Train Data:
    1. put the raw images in hdata/train, put the labeled train data in hdata/ltrain
    2. inside HighDenseTrainPatchMaker.py set the desired number of patches 
       run: python3 HighDenseTrainPatchMaker.py
    3. pltrain and ptrain folders should now contains the relevant patches.
  Prepare Test Data:
    1. put the raw test images in ldata/test, put the labeled test data in ldata/ltest
    2. run: python3 TestPatchMaker.py
    3. ptest should now contains the test patches
    4. inside ValidationPatchMaker.py set the desired number of validation patches
       run: python3 ValidationPatchMaker.py
    5. pvalidation and plvalidation should now contain the relevant patches
  Running main script - pagetrainf8.py
    1. run THEANO_FLAGS=device=cuda0 python3 pagetrainf8.py
    2. measurements of the run are printed at the end
    3. predictions should be inside out folder
    
some useful args inside pagetrainf8.py:
data paths:
  parser.add_argument("--train_images", type = str, default ="hdata/ptrain/"  )
  parser.add_argument("--train_annotations", type = str, default = "hdata/pltrain/"  )
  parser.add_argument("--val_images", type = str , default = "ldata/pvalidation/")
  parser.add_argument("--val_annotations", type = str , default = "ldata/plvalidation/")
  parser.add_argument("--test_images", type = str , default = "ldata/ptest/")
  parser.add_argument("--output_path", type = str , default = "pprediction/")
  
number of classes:
  parser.add_argument("--n_classes", type=int, default = 2 )

patch size:
  parser.add_argument("--input_height", type=int , default = 320  )
  parser.add_argument("--input_width", type=int , default = 320 )
number of epochs:
  parser.add_argument("--epochs", type = int, default = 50 )
  

specific parametes for my run:
data was taken from 'pinkas' data set which contain hebrew subscripts from 18th and 19th centuries.
number of train patches: 30,000
number of validation patches: 1,500
number of epochs: 15 with 1875 iterations each
results:
accuracy level: 0.9635, loss: 0.0877

f-measures:
main text area = 0.915914944576
