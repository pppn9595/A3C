import tensorflow as tf, multiprocessing

WORKER_NUM = 16 #multiprocessing.cpu_count()
REPORT_INTERVAL = 60 * 30 # sec

LOAD_WEIGHTS = False
SEND_MAIL = False

LOAD_FILE = 'save/a3c_save.ckpt'
SAVE_FILE = LOAD_FILE
BACKUP_FILE = 'save/a3c_save_backup.ckpt'

GAME = 'Breakout-v0'
BATCH_SIZE = 512
IMAGE_SIZE = (80, 80) # resize(width, height)
FRAME_NUM = 4
TRAINER = tf.train.AdamOptimizer(learning_rate=0.001) # decrease the learning rate by time (start with ~0.001)

# Mail settings
REPORT_MAIL_SUBJECT = 'Log'
REPORT_MAIL_TO = ''
REPORT_MAIL_FROM_ADDRESS = ''
REPORT_MAIL_FROM_USERNAME = ''
REPORT_MAIL_FROM_PASSWORD = ''
REPORT_MAIL_SMTP = 'smtp.gmail.com:587'