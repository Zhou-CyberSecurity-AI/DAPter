# DAPter
Preventing User Data Abuse in Deep Learning Inference Services

Dataset:
CIFAR-10: 

Link：https://pan.baidu.com/s/1JpB0jyfZ_SMS-A6S5rWhcw  

Password：eTraintv3

Model Result: 

Link：https://pan.baidu.com/s/12bWP7M1R5HjIpqyxOqdwlw

Password：hkba

Abstract:

The data abuse issue has risen along with the widespread development of the deep learning inference service (DLIS). Specifically, mobile users worry about their input data being labeled to secretly train new deep learning models that are unrelated to the DLIS they subscribe to. This unique issue, unlike the privacy problem, is about
the rights of data owners in the context of deep learning. However, preventing data abuse is demanding when considering the usability and generality in the mobile scenario. In this work, we propose, to our best knowledge, the first data abuse prevention mechanism called DAPter. DAPter is a user-side DLIS-input converter, which
removes unnecessary information with respect to the targeted DLIS. The converted input data by DAPter maintains good inference accuracy and is difficult to be labeled manually or automatically for the new model training. DAPter’s conversion is empowered by our lightweight generative model trained with a novel loss function
to minimize abusable information in the input data. Furthermore, adapting DAPter requires no change in the existing DLIS backend and models. We conduct comprehensive experiments with our DAPter prototype on mobile devices and demonstrate that DAPter can substantially raise the bar of the data abuse difficulty with little impact on the service quality and overhead.

DAPter:

![image](https://user-images.githubusercontent.com/35444743/180724855-5fa5507a-54d9-41cb-8cf6-5b54a6a0207f.png)

![image](https://user-images.githubusercontent.com/35444743/180724973-f9b2e08b-e840-40fb-bb42-ec66d55a6cec.png)


Architecture：

![image](https://user-images.githubusercontent.com/35444743/180725016-e79954fb-536f-4d1a-afd2-dc775ad77d0f.png)

Re-implemention

![image](https://user-images.githubusercontent.com/35444743/180725225-9c852f98-9e61-43ca-b5a4-043c62f0ec3f.png)

![image](https://user-images.githubusercontent.com/35444743/180725263-217500d1-48a2-4ad8-91c3-d4c19bc6e211.png)

![image](https://user-images.githubusercontent.com/35444743/180725289-b51cce92-e815-4839-a2fd-517310f7d42a.png)

![image](https://user-images.githubusercontent.com/35444743/180725313-f899866d-1402-4fa6-b6f3-a288f59c78cc.png)



