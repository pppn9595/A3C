import numpy as np, smtplib, tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity

def send_mail(fromaddr, username, password, smtp, to, subject, message):
    message = 'Subject: {}\n\n{}'.format(subject, message)
    server = smtplib.SMTP(smtp)
    server.ehlo()
    server.starttls()
    server.login(username, password)
    server.sendmail(fromaddr, to, str(message))
    server.quit()

def process_frame(frame, size, reshape = True):
    frame = frame[32:, 10:-10] # crop
    frame = resize(frame, size, mode = 'constant')
    frame = rgb2gray(frame)
    frame = rescale_intensity(frame, out_range = (0, 255))
    frame = np.uint8(frame)
    return frame.reshape( (1, *size) ) if reshape else frame
    
def discount_rewards(r, bootstrap = 0., gamma = 0.99):
    r = np.array(r)
    discounted = np.zeros_like(r)
    running_add = bootstrap
    for t in reversed(range(r.size)):
        #if r[t] < 0:
        #    running_add = 0
        running_add = r[t] + running_add * gamma
        discounted[t] = running_add
    return discounted

def start_game(env, frame_num, process_frame_fn, reset = False):
    state = []
    for i in range(frame_num):
        state.append( process_frame_fn( env.reset() if reset and i == 0 else env.step(1)[0], reshape=False) )
    return np.stack(state, axis=0)

def normalized_columns_initializer(std = 1.0):
    def _initializer(shape, dtype = None, partition_info = None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer