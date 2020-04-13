# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import random
import matplotlib.pyplot as plt
from scipy import misc
import time
import pickle

ps = 512  # patch size for training
save_freq = 200
selection_size = 40  # number of photos in selection
epochs_to_train = 2000
amount_of_selections = 60  # number of different selections of images for training
epochs_for_selection = round(epochs_to_train/amount_of_selections)  # epoch to train on one selection

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './results/'+time.strftime('%Y_%m_%d-%H_%M')+'_'+os.path.basename(__file__)+'_'+str(epochs_to_train)+'e_'+str(selection_size)+'i_'+str(epochs_for_selection)+'s/'
result_dir = './results/'+time.strftime('%Y_%m_%d-%H_%M')+'_'+os.path.basename(__file__)+'_'+str(epochs_to_train)+'e_'+str(selection_size)+'i_'+str(epochs_for_selection)+'s/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

# get validation IDs
val_fns = glob.glob(gt_dir + '2*.ARW')
val_ids = [int(os.path.basename(val_fn)[0:5]) for val_fn in val_fns]


DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    input_shortcut = input
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    conv_skip1 = slim.conv2d(input_shortcut, 32, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection1')
    merge1 = tf.concat([conv1, conv_skip1], 3)
    pool1 = slim.max_pool2d(merge1, [2, 2], padding='SAME')

    input_shortcut2 = pool1
    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    conv_skip2 = slim.conv2d(input_shortcut2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection2')
    merge2 = tf.concat([conv2, conv_skip2], 3)
    pool2 = slim.max_pool2d(merge2, [2, 2], padding='SAME')

    input_shortcut3 = pool2
    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    conv_skip3 = slim.conv2d(input_shortcut3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection3')
    merge3 = tf.concat([conv3, conv_skip3], 3)
    pool3 = slim.max_pool2d(merge3, [2, 2], padding='SAME')

    input_shortcut4 = pool3
    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    conv_skip4 = slim.conv2d(input_shortcut4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection4')
    merge4 = tf.concat([conv4, conv_skip4], 3)
    pool4 = slim.max_pool2d(merge4, [2, 2], padding='SAME')

    input_shortcut5 = pool4
    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
    conv_skip5 = slim.conv2d(input_shortcut5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection5')
    merge5 = tf.concat([conv5, conv_skip5], 3)

    # up6 = upsample_and_concat(conv5, conv4, 256, 512)
    up6 = upsample_and_concat(merge5, conv4, 256, 1024)
    input_shortcut6 = up6
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')
    conv_skip6 = slim.conv2d(input_shortcut6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection6')
    merge6 = tf.concat([conv6, conv_skip6], 3)

    # up7 = upsample_and_concat(conv6, conv3, 128, 256)
    up7 = upsample_and_concat(merge6, conv3, 128, 512)
    input_shortcut7 = up7
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')
    conv_skip7 = slim.conv2d(input_shortcut7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection7')
    merge7 = tf.concat([conv7, conv_skip7], 3)

    # up8 = upsample_and_concat(conv7, conv2, 64, 128)
    up8 = upsample_and_concat(merge7, conv2, 64, 256)
    input_shortcut8 = up8
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')
    conv_skip8 = slim.conv2d(input_shortcut8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection8')
    merge8 = tf.concat([conv8, conv_skip8], 3)

    # up9 = upsample_and_concat(conv8, conv1, 32, 64)
    up9 = upsample_and_concat(merge8, conv1, 32, 128)
    input_shortcut9 = up9
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')
    conv_skip9 = slim.conv2d(input_shortcut9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='skip_connection9')
    merge9 = tf.concat([conv9, conv_skip9], 3)

    # conv10 = slim.conv2d(merge9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    # out = tf.depth_to_space(conv10, 2)

    out = slim.conv2d(merge9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    return out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

sess = tf.Session()
# in_image = tf.placeholder(tf.float32, [None, None, None, 4])
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


g_loss = np.zeros((5000, 1))
val_g_loss = np.zeros((5000, 1))

if os.path.exists(checkpoint_dir + "train_loss_list_dump.txt"):
    with open(checkpoint_dir + "train_loss_list_dump.txt", "rb") as dump:
        train_loss_list = pickle.load(dump)
    print("loaded training losses", )
else:
    train_loss_list = []
if os.path.exists(checkpoint_dir + "val_loss_list_dump.txt"):
    with open(checkpoint_dir + "val_loss_list_dump.txt", "rb") as dump:
        val_loss_list = pickle.load(dump)
    print("loaded validation losses")
else:
    val_loss_list = []
if os.path.exists(checkpoint_dir + "training_time_dump.txt"):
    with open(checkpoint_dir + "training_time_dump.txt", "rb") as dump:
        training_time_previous = pickle.load(dump)
    print("loaded training time")
else:
    training_time_previous = 0

allfolders = glob.glob(checkpoint_dir + "*/")
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-5:-1]))

learning_rate = 1e-4

selection_ids = random.sample(train_ids, selection_size)
selection_fns = [None] * len(selection_ids)

first_run = True

# to keep validation images in memory
val_gt_images = [None] * 6000
val_input_images = {}
val_input_images['300'] = [None] * len(val_ids)
val_input_images['250'] = [None] * len(val_ids)
val_input_images['100'] = [None] * len(val_ids)


training_start_time = time.time()
# training
for epoch in range(lastepoch + 1, epochs_to_train + 1):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch > epochs_to_train / 2:
        learning_rate = 1e-5

    # loading new selection
    if (first_run or epoch % epochs_for_selection == 0) and epoch != epochs_to_train:

        first_run = False

        selection_ids = random.sample(train_ids, selection_size)

        gt_images = {}
        gt_images = [None] * 6000
        input_images = {}
        input_images['300'] = [None] * len(selection_ids)
        input_images['250'] = [None] * len(selection_ids)
        input_images['100'] = [None] * len(selection_ids)

        print("Loading new photos to memory")
        st = time.time()

        for ind in np.random.permutation(len(selection_ids)):
            # get the path from image id
            train_id = selection_ids[ind]
            in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
            in_fn = os.path.basename(in_path)

            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            if input_images[str(ratio)[0:3]][ind] is None:
                raw = rawpy.imread(in_path)
                in_raw = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

                gt_raw = rawpy.imread(gt_path)
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

                im = np.expand_dims(np.float32(in_raw / 65535.0), axis=0)
                im = im * np.mean(gt_images[ind]) / np.mean(im)
                input_images[str(ratio)[0:3]][ind] = im


                selection_fns[ind] = in_fn

        print("New photos loaded in %.3f seconds" % (time.time() - st))


    # trainin on every image from selection
    st = time.time()
    for ind in np.random.permutation(len(selection_fns)):
        # get the path from image id
        in_fn = selection_fns[ind]
        train_id = int(os.path.basename(in_fn)[0:5])

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)


        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy:yy + ps, xx:xx + ps, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current

        # print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))


        if epoch % save_freq == 0 or epoch == epochs_to_train:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

            with open(result_dir + "train_loss_list_dump.txt", "wb") as dump:
                pickle.dump(train_loss_list, dump)
            with open(result_dir + "val_loss_list_dump.txt", "wb") as dump:
                pickle.dump(val_loss_list, dump)
            with open(result_dir + "training_time_dump.txt", "wb") as dump:
                pickle.dump(time.time() - training_start_time + training_time_previous, dump)

    print("Epoch: %d Loss=%.3f Time=%.3f" % (epoch, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

    # validate
    # print("\nRunning validation and calculating validation loss")
    st = time.time()
    for ind in np.random.permutation(len(val_ids)):
        val_id = val_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % val_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % val_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        if val_input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            in_raw = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            val_gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            im = np.expand_dims(np.float32(in_raw / 65535.0), axis=0)
            im = im * np.mean(val_gt_images[ind]) / np.mean(im)
            val_input_images[str(ratio)[0:3]][ind] = im

        H = val_input_images[str(ratio)[0:3]][ind].shape[1]
        W = val_input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = val_input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = val_gt_images[ind][:, yy:yy + ps, xx:xx + ps, :]

        input_patch = np.minimum(input_patch, 1.0)

        val_g_loss[ind] = sess.run(G_loss, feed_dict={in_image: input_patch, gt_image: gt_patch})

    print("Epoch: %d Validation_Loss=%.3f Time=%.3f \n" % (epoch, np.mean(val_g_loss[np.where(val_g_loss)]), time.time() - st))

    saver.save(sess, result_dir + 'model.ckpt')

    train_loss_list.append(np.mean(g_loss[np.where(g_loss)]))
    val_loss_list.append(np.mean(val_g_loss[np.where(val_g_loss)]))

# source: https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
training_end_time = time.time()
training_time_this = training_end_time - training_start_time
hours, rem = divmod(training_time_this, 3600)
minutes, seconds = divmod(rem, 60)
print("Training finished in", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

training_time_total = training_time_this + training_time_previous
hours, rem = divmod(training_time_total, 3600)
minutes, seconds = divmod(rem, 60)
print("Total training time", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


with open(result_dir + "train_loss_list_dump.txt", "wb") as dump:
    pickle.dump(train_loss_list, dump)
with open(result_dir + "val_loss_list_dump.txt", "wb") as dump:
    pickle.dump(val_loss_list, dump)
with open(result_dir + "training_time_dump.txt", "wb") as dump:
    pickle.dump(training_time_total, dump)


plt.plot(train_loss_list, label='Training Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(result_dir+'Training and Validation Loss.png', bbox_inches = 'tight')
plt.show()

plt.plot(train_loss_list)
plt.title('Training Loss')
plt.savefig(result_dir+'Training Loss.png', bbox_inches = 'tight')
plt.show()
plt.plot(val_loss_list)
plt.title('Validation Loss')
plt.savefig(result_dir+'Validation Loss.png', bbox_inches = 'tight')
plt.show()

num_of_losses_to_log = round(epochs_to_train/10)
with open(result_dir + "losses.txt", "w") as losses_log:
    for loss in reversed(range(1, num_of_losses_to_log+1)):
        losses_log.write("Training loss: %.3f" % train_loss_list[-loss] + "\t\t" + "Validation loss: %.3f" % val_loss_list[-loss] + "\n")

