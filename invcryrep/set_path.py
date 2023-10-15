
import os,subprocess
import glob



import m3gnet.models
data_path=os.path.dirname(__file__)+'/MP-2021.2.8-EFS'
model_path=m3gnet.models.__path__[0]+'/MP-2021.2.8-EFS/'
subprocess.call(['mkdir','-p', model_path])
subprocess.call(['cp',data_path+'/checkpoint',data_path+'/m3gnet.data-00000-of-00001',\
data_path+'/m3gnet.index',data_path+'/m3gnet.json',model_path])

