# masknomask -Author -mystery2828(Akash C)
A real-time detection of weather a person is wearing mask or not
Steps to follow:
  The uploaded model is not a trained one(it will take up a lot of memory to upload).
  Create a new directory named 'data' and step inside it
  Create directories like  05/01/2020  08:02 PM    <DIR>          record
                           05/01/2020  08:02 PM    <DIR>          test
                           05/01/2020  08:02 PM    <DIR>          train
  In every directory make two folders named 'mask' and 'no-mask'
  
  Now run the recordface.py file and make sure you change the code in the file, i.e., change the path to your file where 'data' is located
  First record for mask for 10-15 seconds and then for no-mask 10-15 seconds.
  Then run the takeroi.py file for both the class(make sure to change the path in the .py file to your path)
  Then run the train.py file(make sure to make any changes in the ImageDataGeneratoe part according to the comments)
  And yeah your model is trained now!!!!!!!!
  Run the predict.py file and your realtime detection is done.(FPS depends on the system config).
