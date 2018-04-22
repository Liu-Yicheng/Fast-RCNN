# 工程内容
这个程序是基于tensorflow实现Fast-RCNN功能。

# 开发环境
Ubuntu16.04（i5-7500 + GTX 1070Ti ） + python3.5 + tensorflow1.3 +  cv2 +  roi-pooling

roi-pooling ：是需要额外安装的op        
　github地址为：https://github.com/deepsense-ai/roi-pooling    
　安装的必须条件：linux系统 + GPU     
　安装过程的坑：注意makefile文件中cuda的安装路径是否与自己电脑上的安装路径一致    
　　　　　　　　tensorflow1.4以上版本还可能涉及到动态库的问题，不过网上都有解决方案     

# 数据集
通用数据集：   
	　数据集来源：北京理工大学BIT车辆数据集（1万张照片、6类车辆）   
	　数据集数量：从1万张中选取了2100张，每一类350张。     
	　数据集制作：由于数据源没有带xml文件，带了mat文件，所以重新制作了PASCAL VOC格式的xml标注文件。    
	　数据集下载地址：https://pan.baidu.com/s/1X-8E5eGldAfTHdyJXlFllA   密码：ivq8    

自己用的数据集：接触网6C成像图（由于在车数据集上的实验结果被误删了、所以下面用此数据集的实验结果展示）

# 程序说明   
1、config.py---网络定义、训练与数据处理所需要用到的参数        
2、Net.py---用于定义Alexnet_Net模型     
4、Data.py---用于处理数据的各种方法     
5、train.py---用于各类模型的训练与测试、主函数      
6、selectivesearch.py---选择性搜索代码      


# 文件夹说明
1、Data：   
	　Annotations--存放图片标注的xml文件（手动存放）  
	　Images --存放用于训练与测试的图片（手动存放）  
	　Processes --存放处理xml文件之后形成图片label信息的npy文件（程序执行）  
	　all_list.txt --存放图片序号 用于处理与序号所对应的xml文件，以形成label信息的npy文件（手动存放）   
	　test_list.txt --存放测试图片序号（手动存放）    
	　train_list.txt --存放训练图片序号（手动存放）    
2、Output --存放训练过程的config文件、log文件、weight文件（程序执行）   
3、Test_output 图片 -- 存放测试图片的测试结果（程序执行）    
4、train_alexnet        
　　存放Alexnet的在Imagenet上训练好的权重，用这个权重来finetune （手动存放）   

# 实验结果展示：
自己用的数据集：接触网6C成像图（由于在车数据集上的实验结果被误删了、所以下面用此数据集的实验结果展示   
（只是为了验证自己写的程序没有错误，并未经过十分深入的调参，所以结果的准确度应该可用再提升）  

检测结果：   
![result_2](https://github.com/Liu-Yicheng/Fast-RCNN/raw/master/result/2.jpg)  

特征层可视化结果（第五个卷积层经过relu后第49张特征图）： 
![result_1](https://github.com/Liu-Yicheng/Fast-RCNN/raw/master/result/1.jpg)   
![result_3](https://github.com/Liu-Yicheng/Fast-RCNN/raw/master/result/3.jpg)   



# 程序问题
1.由于此次的程序大体上都是自己编写，代码或许不太健壮。在自己的环境下运行过没有问题，如在其他环境下不能运行应该只需要微调下   

2.程序使用流程：    
　　　　---------------------训练过程---------------------------        
　　　　A.将需要训练与测试的图片放入Data/Images文件夹，将XML文件放入Data/Annotation文件夹    
　　　　B.将要训练与测试图片的编号写入all_list,训练图片的编号写入train_list，测试图片的编号写入test_list    
　　　　C.下载Alexnet预训练权重，放入Alexnet_weight文件夹。    
　　　　　权重下载地址：https://pan.baidu.com/s/1XhEpG_dNeUlnegH4zYxgrw  密码：l7um    
　　　　D.针对自己的项目修改config文件中的参数   
　　　　E.将train.py中main函数的train改为True，开始训练    
　　　　----------------------测试过程---------------------------    
　　　　A.修改config中的weight_file改为你训练输出的权重文件地址    
　　　　B.修改train.py文件第29行为：
　　　　　　self.variable_to_restore = slim.get_variables_to_restore((exclude=[])    
　　　　C.将train.py中main函数的train改为False，开始测试   
    
3.Fast_RCNN检测的准确度与提取的候选框质量有很大的关系、因此提取候选框的算法至关重要。  
　在我的数据集上selectivesearch提取候选框的效果并不好所以自己根据自己数据集的特点重新   
　写了一个候选框提取的算法，以此达到了还算不错的效果。这里还是放上了selectivesearch的提取算法。       

	
				
				


