import os
import random
import sys
#sys.argv =["", "NWPU", "WHU-SAR"]
# sys.argv =["", "AID", "SAR"]
sys.argv =["", "MLRSNet", "WHU-SAR"]

source = sys.argv[1]
target = sys.argv[2]

#p_path = os.path.join('/data/zxf/MLRSNet_dataset_1/')
p_path = os.path.join('/data2/yaoxiwen4/dataset/MLRSNet_6/')
#p_path = os.path.join('/data/zxf/WHU-SAR6/')

dir_list = os.listdir(p_path)
#print(dir_list)
#class_list_shared = ["Airfield", "Anchorage", "Beach", "Farm", "Flyover", "Forest",  "Parkingspace", "River", "Sparseresidential"]
class_list_shared = ["bridge", "building", "farmland",  "lake", "mountain", "river"]
# class_list_shared = ["airplane", "bridge", "building", "farmland", "harbor",  "parking_lot", "ship", "storage_tank"]
# class_list_shared = ["airplane", "bridge", "building", "farmland", "harbor",  "ship", "storage_tank"]
#class_list_shared = ["airplane", "bridge", "building", "farmland", "harbor", "storage_tank"]


#unshared_list = list(set(dir_list) - set(class_list_shared))
print(class_list_shared)
#unshared_list.sort()
#print(unshared_list)
#source_list = class_list_shared + unshared_list[:3]
source_list = class_list_shared
print(source_list)
#private_t = list(set(unshared_list)- set(source_list))
#private_t = list(set(unshared_list))

#print(private_t)
target_list = class_list_shared
print(target_list)
path_source = "/data2/yaoxiwen4/home/disk1/code_1/ATM/data_rs/MLRSNet2WHU/source_%s.txt"%(source)
path_target = "/data2/yaoxiwen4/home/disk1/code_1/ATM/data_rs/MLRSNet2WHU/target_%s.txt"%(target)
write_source = open(path_source,"w")
write_target = open(path_target,"w")
dir_list.sort()
#dir_list.sort()
for k, direc in enumerate(dir_list):

    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        figList = []  # 保存所有图片的前缀，且是整型
        for fig in files:
            if fig.endswith('.jpg'):
                pre, figType = fig.split('.')  # 前缀
                # pre_1, pre_2 = pre.split('_')  # 前缀
                pre = pre.split('_')  # 前缀
                pre_1 = pre[0]
                if len(pre) > 2:
                    figList.append(int(pre[2]))
                else:
                    figList.append(int(pre[1]))  # 获取图片名称前缀，且转化为 整型
                # figList.append(int(pre))  # 获取图片名称前缀，且转化为 整型
        figList.sort()  # 图片的前缀的整型 形式排序， 这样应该就是 0 1 2 3排序，而不是 0 1 101 102 这样子
        filesList = []
        for figInt in figList:
            fig = str(figInt)
            if len(pre) > 2:
                fig_1 = pre_1 + '_' + pre[1] + '_' + fig + '.jpg'
            else:
                fig_1 = pre_1 + '_' + fig + '.jpg'
            filesList.append(fig_1)


        # files = os.listdir(os.path.join(p_path, direc))
        # files.sort()
        for i, file in enumerate(filesList):
            #class_list_shared.sort()
            if direc in class_list_shared:
            #if direc in source_list:

                class_name = direc

                #file_name = os.path.join('/home/zxf/disk1/zxf/code_1/DANCE/office-31', source, 'images2', direc, file)
                #file_name = os.path.join('/home/zxf/disk1/zxf/code_1/DANCE/dataset', source, 'images3', direc, file)
                file_name = os.path.join('/data2/yaoxiwen4/dataset/MLRSNet_6/', direc, file)
                #file_name = os.path.join('/data/zxf/MLRSNet_dataset/', direc, file)
                #file_name = os.path.join('/data/zxf/WHU-SAR6/', direc, file)
                write_source.write('%s %s\n' % (file_name, class_list_shared.index(class_name)))
                #write_source.write('%s %s\n' % (file_name, source_list.index(class_name)))
            else:
                continue
#p_path = os.path.join('/home/zxf/disk1/zxf/code_1/DANCE/office-31', target,'images')

#p_path = os.path.join('/home/zxf/disk1/zxf/code_1/DANCE/dataset', target,'images2')
p_path = os.path.join('/data2/yaoxiwen4/dataset/WHU-SAR6/')
#p_path = os.path.join('/data/zxf/NWPU-SAR6/')

#p_path = os.path.join('/data/zxf/MLRSNet_dataset/')
#p_path = os.path.join('/data/zxf/NWPU-45/')

#p_path = os.path.join('/data/zxf/AID/')

dir_list = os.listdir(p_path)
dir_list.sort()
#print(dir_list)
for k, direc in enumerate(target_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        figList = []  # 保存所有图片的前缀，且是整型
        for fig in files:
            if fig.endswith('.jpg'):
                pre, figType = fig.split('.')  # 前缀
                # pre_1, pre_2 = pre.split('_')  # 前缀
                pre = pre.split('_')  # 前缀
                pre_1 = pre[0]
                if len(pre) > 2:
                    figList.append(int(pre[2]))
                else:
                    figList.append(int(pre[1])) # 获取图片名称前缀，且转化为 整型
                #figList.append(int(pre))  # 获取图片名称前缀，且转化为 整型
        figList.sort() # 图片的前缀的整型 形式排序， 这样应该就是 0 1 2 3排序，而不是 0 1 101 102 这样子
        filesList = []
        for figInt in figList:
            fig = str(figInt)
            if len(pre) > 2:
                fig_1 = pre_1 + '_' + pre[1] + '_' + fig + '.jpg'
            else:
                fig_1 = pre_1 + '_' + fig + '.jpg'
            filesList.append(fig_1)

        #files.sort()
        #for i, file in enumerate(files):
        for i, file in enumerate(filesList):
            #file_name = os.path.join('/home/zxf/disk1/zxf/code_1/DANCE/office-31', target, 'images', direc, file)
            #file_name = os.path.join('/home/zxf/disk1/zxf/code_1/DANCE/dataset', target, 'images2', direc, file)
            file_name = os.path.join('/data2/yaoxiwen4/dataset/WHU-SAR6/', direc, file)
            #file_name = os.path.join('/data/zxf/NWPU-SAR6/', direc, file)
            # file_name = os.path.join('/data/zxf/MLRSNet_dataset/', direc, file)
            #file_name = os.path.join('/data/zxf/NWPU-45/', direc, file)
            #file_name = os.path.join('/data/zxf/AID/', direc, file)

            if direc in class_list_shared:
            #if direc in source_list:
                class_name = direc
                write_target.write('%s %s\n' % (file_name, class_list_shared.index(class_name)))
                #write_target.write('%s %s\n' % (file_name, source_list.index(class_name)))
            elif direc in target_list:
                file_name = os.path.join(p_path, direc, file)
                write_target.write('%s %s\n' % (file_name, len(class_list_shared)))
                write_target.write('%s %s\n' % (file_name, len(source_list)))






