# MTCNN
Contains the implementation of the MTCNN paper excluding face key points.

- Tensorflow 2.1.0
- opencv 4.2.0
- python3


# Testing
- To detect the faces from an input image,use following command:
  
  python3 detect_face.py \<input image path\> \<output image path\>

# Sample Result
<img src="https://github.com/gurushantj/MTCNN/blob/master/results/bharat_natyam_output.jpg?raw=true" alt="Smiley face" height="520" width="820">

<img src="https://github.com/gurushantj/MTCNN/blob/master/results/cricket_output.jpg?raw=true" alt="Smiley face" height="520" width="820">

# Training
This model is trained from scratch without using any pre trained weights , if you wish you can initilize with the existing weights and train on your dataset.
- python3 data_gen/gen_shuffle_data.py <size> # size could be 12 or 24 or 48 based upon the network to train
- Run python3 data_gen/CreateTFRecordTraining.py # to generate the tf record file each containg 200k records
- Go to main.py and scroll down to bottom and uncomment the fun call to run the training on(PNet/RNet/ONet).

# References
<table>
  <tr>
    <th>References</th>
    <th>Modified</th>
  </tr>
  <tr>
    <td>https://arxiv.org/pdf/1604.02878</td>
    <td>No</td>
  </tr>
  <tr>
    <td>https://github.com/wangbm/MTCNN-Tensorflow</td>
    <td>Modified</td>
  </tr>
</table>
