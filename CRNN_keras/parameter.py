import sys
CHAR_VECTOR = "adefghjknqrstwABCDEFGHIJKLMNOPZ0123456789"
#import string
#characters=string.digits+string.ascii_uppercase+string.ascii_lowercase
print(sys.path[0])
f1 = open(sys.path[0]+"/1.txt","r")
character = f1.readlines()
character=''.join(character)
characters=character.replace("\n", "")
letters=str(characters)+' '
#letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1

img_w, img_h = 256, 32

# Network parameters
batch_size =256
val_batch_size = 16

downsample_factor = 4
max_text_len =16