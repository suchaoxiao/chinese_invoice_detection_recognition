import tornado.ioloop
import tornado.web
import base64
import os,sys
from test import test
import session_lei
import tensorflow as tf
from tensorflow.python.platform import gfile
from config import crnn_modelpath
from CRNN_Keras.model_crnn import get_model


py_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(py_dir)
sys.path.append(project_dir)
upload_path="/invoice/upload/"
file_dir_img_root = project_dir+upload_path

env = sys.argv[0]
container = {}
if env=='production':
    file_dir_img_root = "upload/"
#logger.debug(file_dir_img_root)
if not os.path.exists(file_dir_img_root):
    os.makedirs(file_dir_img_root)




def clean_file(filename):
    return
    #logger.debug(filename)
    #logger.debug(os.path.join(file_dir_img_root, filename + '.jpg'))
    #logger.debug(os.path.join(file_dir_img_root, filename))
    if os.path.exists(os.path.join(file_dir_img_root,filename+'.jpg')):
        os.remove(os.path.join(file_dir_img_root,filename+'.jpg'))
    if os.path.exists(os.path.join(file_dir_img_root,filename)):
        shutil.rmtree(os.path.join(file_dir_img_root,filename))


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        session = session_lei.Session(self, 1)
        if session['session'] != True:  # 判断session里的zhuangtai等于True
            self.redirect("/login")
        self.write('''
<html>
  <head><title>Upload File</title></head>
  <body>
    <form action='/invoice/index' enctype="multipart/form-data" method='post'>
    <input type='file' name='提交文件'/><br/>
    <input type='submit' value='单张识别'/>
    </form>
  </body>
</html>
''')
    def get_base64(self,file):
        data = None
        with open(file, 'rb') as f:
            data = base64.b64encode(f.read())  # 读取文件内容，转换为base64编码
        return data


    def post(self):

        upload_path=os.path.join(os.path.dirname(__file__),'upload')  #文件的暂存路径

        file_metas=self.request.files['提交文件']    #提取表单中‘name’为‘file’的文件元数据
        for meta in file_metas:
           # filename=str(uuid.uuid1())
            #filepath=os.path.join(upload_path,filename+'.jpg')
            #print(str(meta))
            filename = meta['filename']
            filepath = os.path.join(upload_path, filename)
            with open(filepath,'wb') as up:      #有些文件需要已二进制的形式存储，实际中可以更改
                up.write(meta['body'])
            #result = img.img_process(filepath,config)
            result=test(filepath,sess,output_cls_prob,output_box_pred,input_img,keras_model)
            file_path_src=filepath
            img_data = {}
            log = {}
            img_data["img_src"] = self.get_base64(upload_path + '/crop_img/' + filename)
            img_data["src"] = self.get_base64(file_path_src)
            for index in range(len(result['file'])):

                img_data['buildfile_'+str(index)]=self.get_base64(result['file'][index])
                log['machine_default_'+str(index)]=result['result'][index]

            clean_file(filename)
            self.render("templates/demo.html", img_data=img_data,log=log)
'''            
class ALLIndexHandler(tornado.web.RequestHandler):
    def get(self):
        hq_cookie = self.get_cookie('xr_cookie')  # 获取浏览器cookie
        session = container.get(hq_cookie, None)  # 将获取到的cookie值作为下标，在数据字典里找到对应的用户信息字典
        if not session:  # 判断用户信息不存在
            self.render("demo.html")  # 显示登录html文件          #打开到登录页面
        else:
            if session.get('is_login', None) == True:  # 否则判断用户信息字典里的下标is_login是否等于True
                self.redirect("/invoice/ALLindex")  # 显示index.html文件    #跳转查看页面
            else:
                self.redirect("/login")
    def get_base64(self,file):
        data = None
        with open(file, 'rb') as f:
            data = base64.b64encode(f.read())  # 读取文件内容，转换为base64编码
        return data


    def post(self):

        upload_path=os.path.join(os.path.dirname(__file__),'upload/origin')  #文件的暂存路径
        lists = os.listdir(upload_path)
        for list in lists:
           # filename=str(uuid.uuid1())
            filepath=os.path.join(upload_path,list+'.jpg')
            #print(str(meta))

            #result = img.img_process(filepath,config)
            result=test(filepath)
            file_path_src=list
            engineno_buildfile=result['file'][0]
            cardno_buildfile=result['file'][1]
            vinno_buildfile=result['file'][2]
            price_buildfile=result['file'][3]
            img_data={}
            img_data["src"]=self.get_base64(upload_path+'/crop_img/'+filepath)
            img_data["img_src"] = self.get_base64(file_path_src)
            if cardno_buildfile != None and os.path.exists(cardno_buildfile):
                img_data["cardno_buildfile"] = self.get_base64(cardno_buildfile)
            if engineno_buildfile != None and os.path.exists(engineno_buildfile):
                img_data["engineno_buildfile"] = self.get_base64(engineno_buildfile)
            if vinno_buildfile != None and os.path.exists(vinno_buildfile):
                img_data["vinno_buildfile"] = self.get_base64(vinno_buildfile)
            if price_buildfile != None and os.path.exists(price_buildfile):
                img_data["price_buildfile"] = self.get_base64(price_buildfile)

            log={}
            log["cardno_machine_default"]=result['result'][0]
            log["engineno_machine_default"] =result['result'][1]
            log["vinno_machine_default"] = result['result'][2]
            log["price_machine_default"] = result['result'][3]

'''
class LoginHandler(tornado.web.RequestHandler):
    def get(self):
        session = session_lei.Session(self, 1)  # 创建session对象，cookie保留1天
        if session['session'] == True:  # 判断session里的zhuangtai等于True
            self.redirect("/invoice/index")  # 跳转到查看页面
        else:
            self.render("templates/login.html", tishi='请登录')  # 打开登录页面


    def post(self):
        username = self.get_body_argument('username', None)
        password = self.get_body_argument('password', None)
        if username == 'admin' and password == 'admin':  # 判断用户名和密码
            sessions = session_lei.Session(self, 1)  # 创建session对象，cookie保留1天
            sessions['username'] = username  # 将用户名保存到session
            sessions['password'] = password  # 将密码保存到session
            sessions['session'] = True  # 在session写入登录状态
            self.redirect("/invoice/index")  # 跳转查看页面
        else:
            self.render("templates/login.html", tishi='用户名或密码错误')  # 打开登录页面
application=tornado.web.Application([
    (r'/invoice/index',IndexHandler),
   # (r'/invoice/ALLindex',ALLIndexHandler),
    (r'/login', LoginHandler)
],debug=True)

def load_model():
    modelPath = crnn_modelpath

    model = get_model(training=False)

    try:
        model.load_weights(modelPath)
        print("...Previous weight data...")
    except:
        raise Exception("No weight file!")
    return model
if __name__=="__main__":
    application.listen(7777)
    config = tf.ConfigProto(allow_soft_placement=True)

    sess = tf.Session(config=config)
    with gfile.FastGFile('text-detection-ctpn-master/data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
    keras_model=load_model()

    tornado.ioloop.IOLoop.instance().start()