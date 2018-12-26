import cv2
import os, random
import numpy as np
from parameter import letters,characters,max_text_len

# # Input data generator
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(letters.find(char))
    return ret
def get_label(text):
    # print(str(os.path.split(filepath)[-1]).split('.')[0].split('_')[-1])
    lab=[]
    for num in str(text):
        lab.append(int(characters.find(num)))
    if len(lab) < max_text_len:
        cur_seq_len = len(lab)
        for i in range(max_text_len - cur_seq_len):
            print
            lab.append(len(characters)+1) #
    return lab

class TextImageGenerator:
    def __init__(self, dir_path, label_dirpath, img_w, img_h,
                 batch_size, downsample_factor):
        self.texts = []
        self.img_dir = []
        f1 = open(label_dirpath, "r")
        label_list = f1.readlines()

        for index in range(int(len(label_list) / 2)):

            file = label_list[2 * index]
            file = file.replace("\n", "")
            label = label_list[2 * index + 1]
            label = label.replace("\n", "")


            self.img_dir.append(file)
            self.texts.append(label)
        self.dir_path=dir_path
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        #self.img_dirpath = img_dirpath  # image dir path
        # self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = len(self.img_dir)  # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        # self.texts = []

        ## samples의 이미지 목록들을 opencv로 읽어 저장하기, texts에는 label 저장

    def build_data(self):
        print(self.n, " Image Loading start...")
        for i, img_file in enumerate(self.img_dir):
            img = cv2.imread(self.dir_path+img_file, cv2.IMREAD_GRAYSCALE)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0

            self.imgs[i, :, :] = img

        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):  ## index max -> 0 으로 만들기
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.ones([self.batch_size, self.img_h, self.img_w, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, max_text_len]) * -1           # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 1)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                #img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                label_length[i] = len(text)
                Y_data[i, 0:len(text)] = text_to_labels(text)
            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)