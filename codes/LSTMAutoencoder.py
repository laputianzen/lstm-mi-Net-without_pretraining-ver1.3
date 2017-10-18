
import tensorflow as tf
#from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.contrib.rnn import LSTMCell

import numpy as np
import os
import time
from os.path import join as pjoin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import pickle as pl


"""
Future : Modularization
"""

class LSTMAutoencoder(object):
  """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)

  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

  def __init__(self, hidden_num, inputs, seqlen, #max_x, max_y,
    cell=None, optimizer=None, reverse=True, 
    decode_without_input=False):
    """
    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size 
              (batch_num x elem_num)
      cell : an rnn cell object (the default option 
            is `tf.python.ops.rnn_cell.LSTMCell`)
      optimizer : optimizer for rnn (the default option is
              `tf.train.AdamOptimizer`)
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
    """

    self.max_step_num = int(inputs.shape[0]) 
    #self.batch_num = inputs[0].get_shape().as_list()[0]
    self.elem_num = inputs[0].get_shape().as_list()[1]   
    self.input_ = inputs
    
    sequence_mask_queue = []
    for i in range(self.elem_num):
        sequence_mask = tf.sequence_mask(seqlen,self.max_step_num)
        sequence_mask_queue.append(sequence_mask)
    seqMask = tf.stack(sequence_mask_queue,axis=0)
    self.seqMask = tf.cast(tf.transpose(seqMask,perm=(2,1,0)),tf.float32)
# =============================================================================
#     'extra setting'
#     self.seqlen = seqlen
#     self.p_input = tf.transpose(inputs, perm=[1,0,2])
# =============================================================================
    
    if cell is None:
      self._enc_cell = LSTMCell(hidden_num)
      self._dec_cell = self._enc_cell
      #self._dec_cell = LSTMCell(hidden_num)
    else :
      self._enc_cell = cell
      self._dec_cell = cell

    with tf.variable_scope('encoder'):
      self.z_codes, self.enc_state = tf.nn.dynamic_rnn(
        self._enc_cell, inputs, dtype=tf.float32, time_major=True,
        sequence_length=seqlen) 
        
    with tf.variable_scope('decoder') as vs:
      dec_weight_ = tf.Variable(
        tf.truncated_normal([hidden_num, self.elem_num], dtype=tf.float32),
        name="dec_weight")
#      dec_bias_ = tf.Variable(
#        tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32),
#        name="dec_bias")

      if decode_without_input:
        #dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32) 
        #              for _ in range(inputs.shape[0])]
        #              #for _ in range(len(inputs))]
        #dec_inputs = tf.zeros(inputs.shape)
        dec_inputs = tf.zeros_like(inputs)
        dec_outputs, dec_state = tf.nn.dynamic_rnn(
          self._dec_cell, dec_inputs,sequence_length=seqlen,
          initial_state=self.enc_state, dtype=tf.float32, time_major=True)
        """the shape of each tensor
          dec_output_ : (step_num x hidden_num)
          dec_weight_ : (hidden_num x elem_num)
          dec_bias_ : (elem_num)
          output_ : (step_num x elem_num)
          input_ : (step_num x elem_num)
        """
        if reverse:
          #dec_outputs = dec_outputs[::-1]
          dec_outputs = tf.reverse_sequence(dec_outputs,seqlen,seq_axis=0,batch_axis=1)
        dec_output_ = tf.reshape(dec_outputs, [-1, hidden_num])
        #dec_output_ = tf.transpose(tf.pack(dec_outputs), [1,0,2])
        #dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [self.batch_num,1,1])
        #self.output_ = tf.batch_matmul(dec_output_, dec_weight_) + dec_bias_
        self.output_ = tf.matmul(dec_output_, dec_weight_) #+ dec_bias_
        #self.output_ = tf.reshape(self.output_,inputs.shape)
        self.output_ = tf.reshape(self.output_,[self.max_step_num ,-1,self.elem_num])
      else : 
        dec_state = self.enc_state
        dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
        dec_outputs = []
        #for step in range(inputs.shape[0]):
        for step in range(seqlen):
          if step>0: vs.reuse_variables()
          dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
          dec_input_ = tf.matmul(dec_input_, dec_weight_) #+ dec_bias_
          dec_outputs.append(dec_input_)
        if reverse:
          #dec_outputs = dec_outputs[::-1]
          dec_outputs = tf.reverse_sequence(dec_outputs,seqlen,seq_axis=0,batch_axis=1)
        #self.output_ = tf.transpose(tf.pack(dec_outputs), [1,0,2])
        self.output_ = tf.stack(dec_outputs)

    #self.input_ = tf.transpose(tf.pack(inputs), [1,0,2])
    
    self.last_output, self.index, self.flat = self._last_relevant(self.z_codes, seqlen)
    #self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))
# =============================================================================
#     ball_dim_x = tf.cast(tf.ones_like(sequence_mask_queue[0]),tf.float32) * max_x
#     ball_dim_y = tf.cast(tf.ones_like(sequence_mask_queue[0]),tf.float32) * max_y
#     ball_dim = tf.stack([ball_dim_x,ball_dim_y], axis=0)
#     self.ball_dim = tf.transpose(ball_dim,perm=(2,1,0))
# =============================================================================
    self.difference = (self.input_ - self.output_) #*self.ball_dim
    self.exact_difference = tf.multiply(self.difference,self.seqMask)  
    self.l2_norm = tf.norm(self.exact_difference,axis=2)
    #self.loss = tf.reduce_sum(tf.norm(self.exact_difference,axis=2))
    self.loss = tf.reduce_mean(tf.square(self.exact_difference))
    #self.loss = tf.reduce_mean(tf.norm((self.input_ - self.output_),axis=2))
    self.output = tf.transpose(self.output_, perm=[1,0,2])
    if optimizer is None :
        self.train = tf.train.AdamOptimizer().minimize(self.loss)
      #self.train = tf.train.RMSPropOptimizer(0.5).minimize(self.loss)
    else :
        self.train = optimizer.minimize(self.loss)
  
  @staticmethod
  def _last_relevant(output, length):
      max_length = int(output.get_shape()[0])
      batch_size = tf.shape(output)[1]
      output_size = int(output.get_shape()[2])
      #index = tf.range(0, batch_size) * max_length + (length - 1)
      index = batch_size * (length - 1) + tf.range(0, batch_size)
      flat = tf.reshape(output, [-1, output_size])
      relevant = tf.gather(flat, index)
      relevant = tf.reshape(relevant,[batch_size,output_size])
      return relevant, index, flat
  
#  @property
#  def length(self):
#      used = tf.sign(tf.reduce_max(tf.abs(self.input_), reduction_indices=2))
#      length = tf.reduce_sum(used, reduction_indices=1)
#      length = tf.cast(length, tf.int32)
#      return length      

def pretraining(ae,sess,dataset,FLAGS):
    """
    pre-training cycle
    """
   
#    loss_queue=[]
#    outputs = []
    
    vars_to_init = tf.trainable_variables()#tf.global_variables()  
    hist_summaries = [tf.summary.histogram(v.op.name + "_lstm_pretraining", v)for v in vars_to_init]
    summary_op = tf.summary.merge(hist_summaries)     
    
    pretrain_train_loss  = tf.summary.scalar('pretrain_train_loss',tf.sqrt(ae.loss)*FLAGS.MAX_Y)
    pretrain_test_loss  = tf.summary.scalar('pretrain_test_loss',tf.sqrt(ae.loss) *FLAGS.MAX_Y)
#    fold = FLAGS.fold
    'lstm cell parameter'
#    hidden_num = FLAGS.lstm_hidden_dim
    elem_num = FLAGS.lstm_input_dim
#    activation_lstm_name = FLAGS.lstm_activation
    max_step_num = FLAGS.lstm_max_sequence_length
    iteration = FLAGS.lstm_pretrain_iteration
    pretrain_batch_size = FLAGS.lstm_pretrain_batch_size
    
    'lstm placeholders'
    #p_input = FLAGS.lstm.p_input
    #seqlen = FLAGS.lstm.seqlen
    p_input = FLAGS.p_input
    seqlen = FLAGS.seqlen
    
    'dataset settings'
    num_train = dataset.num_train
    num_test  = dataset.num_test
    numPlayer = dataset.numPlayer
    padS = dataset.dataTraj

    seqLenMatrix = dataset.seqLenMatrix
    testIdx_fold = dataset.testIdx
    trainIdx_fold= dataset.trainIdx
    
    saver = tf.train.Saver(vars_to_init,max_to_keep=int(iteration/FLAGS.ae_lstm_save_ckpt_step+1))
#    _pretrain_model_dir = '{0}/fold{1}/ae_{5}/{3}/hidden{2}/iter{4}'.format(
#            FLAGS._ckpt_dir,fold+1,hidden_num,activation_lstm_name,iteration, FLAGS.lstm_type)
    _pretrain_model_dir = FLAGS.ae_lstm_pretrain_model_dir
    if not os.path.exists(_pretrain_model_dir):
        os.makedirs(_pretrain_model_dir)
    #FLAGS._lstm_pretrain_model_dir = _pretrain_model_dir
    model_ckpt = _pretrain_model_dir + '/model.ckpt-{0}'.format(iteration)
      
    if not FLAGS.auto_load_ckpt:
        if os.path.isfile(_pretrain_model_dir+'/checkpoint'):
            step = input(">>> Input: ")
            model_ckpt = model_ckpt + '-{0}'.format(step)
            if step == "" or not(os.path.isfile(model_ckpt+'.meta')):
                print('step ({0}) not existed!!'.format(step))           
                model_ckpt= tf.train.latest_checkpoint(_pretrain_model_dir)
                print('load lastest checkpoint: {0} instead ...'.format(model_ckpt))
            else:
                print('load checkpoint: {0}'.format(model_ckpt))
    else:
        model_ckpt= tf.train.latest_checkpoint(_pretrain_model_dir)
        print('load checkpoint: {0}'.format(model_ckpt))
#    else:
#        model_ckpt = _pretrain_model_dir  + '/model.ckpt-{0}'.format(step)
    
    if os.path.isfile(model_ckpt+'.meta'):
        #tf.reset_default_graph()
        print("|---------------|---------------|---------|----------|")
        saver.restore(sess, model_ckpt)
        for v in vars_to_init:
            print("%s with value %s" % (v.name, sess.run(tf.is_variable_initialized(v))))
        
        #check model loaded generate good result
        random_sequences = np.reshape(padS[testIdx_fold,:],(-1,max_step_num,elem_num))
        batch_seqlen = np.reshape(seqLenMatrix[testIdx_fold,:],(-1))
        test_loss = sess.run(ae.loss,  feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
        print("test loss: %f" %(np.sqrt(test_loss)*FLAGS.MAX_Y))
    else:
        sess.run(tf.global_variables_initializer())
        if FLAGS.ae_lstm_debug:
            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            print(values)
        summary_writer = tf.summary.FileWriter(pjoin(FLAGS.ae_lstm_summary_dir,
                                  'lstm_pre_training'),graph=tf.get_default_graph(),flush_secs=FLAGS.flush_secs)
        for i in range(iteration):
            start_time = time.time()
            perm = np.arange(num_train)
            np.random.shuffle(perm)
    
          
            train_loss = 0.0
            for v in range(int(num_train/pretrain_batch_size)): # one video at a time
#                player_perm = np.arange(numPlayer)
#                np.random.shuffle(player_perm)
#                random_sequences = padS[perm[v],player_perm]
#                batch_seqlen = np.reshape(seqLenMatrix[perm[v],:],(-1))
                vv = perm[pretrain_batch_size*v:pretrain_batch_size*v+pretrain_batch_size]
                v_order = np.arange(pretrain_batch_size * numPlayer)
                np.random.shuffle(v_order)
                X = padS[vv]
                X1 = np.reshape(X,(-1,X.shape[2],X.shape[3]))
                
                random_sequences = X1[v_order]
                batch_seqlen = np.reshape(seqLenMatrix[vv,:],(-1))
                batch_seqlen= batch_seqlen[v_order]
                
                loss_val, _= sess.run([ae.loss, ae.train],  
                                       feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
                #if np.sum(output - output_seq[batch_seqlen[0]-1]):
                if FLAGS.ae_lstm_debug:
                    l2_norm,exact_diff, diff, bm, dec_val, output, output_seq, index, flat = sess.run([
                            ae.l2_norm,ae.exact_difference,ae.difference, ae.seqMask, ae.output,
                            ae.last_output, ae.z_codes, ae.index, ae.flat],feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
                    if np.sum(output - output_seq[[batch_seqlen-1,np.arange(pretrain_batch_size * numPlayer)]]):
                        raise AssertionError("two output are not equal")
                        
                    np.savetxt('tmp/lstm{}_{}.txt'.format(i,v),output,fmt='%.5f', delimiter=' ')
                    #outputs.append(output)
                #print('scale difference', np.sqrt(loss_val)*FLAGS.MAX_Y)
                train_loss += loss_val
                
                if v % FLAGS.ae_lstm_save_summary_step == 0 or v == num_train/pretrain_batch_size-1: #10
                    summary_str = sess.run(summary_op, feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
                    summary_writer.add_summary(summary_str, i*(num_train/pretrain_batch_size)+v)
                    #print("iter %d, vid %d: %f" % (i+1, v+1, loss_val))
                
                
            
            if i % FLAGS.ae_lstm_save_ckpt_step == 0 or i == iteration-1: #10
                save_path = saver.save(sess, model_ckpt,global_step=i)
                print('save model ckeck point', save_path)
            print("iter %d:" %(i+1))
            print("train loss: %f" %(np.sqrt(train_loss*pretrain_batch_size/num_train)*FLAGS.MAX_Y))        
          
            'train data average loss'
            random_sequences = np.reshape(padS[trainIdx_fold,:],(-1,max_step_num,elem_num))
            batch_seqlen = np.reshape(seqLenMatrix[trainIdx_fold,:],(-1))
            summary_train_loss_str = sess.run(pretrain_train_loss, feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
            summary_writer.add_summary(summary_train_loss_str, i)
            #test_loss = 0.0
            #for v in range(num_test):
            'test data average loss'
            random_sequences = np.reshape(padS[testIdx_fold,:],(-1,max_step_num,elem_num))
            batch_seqlen = np.reshape(seqLenMatrix[testIdx_fold,:],(-1))
            dec_val,test_loss = sess.run([ae.output,ae.loss], feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
            
            summary_test_loss_str = sess.run(pretrain_test_loss, feed_dict={p_input:random_sequences,seqlen: batch_seqlen})
            summary_writer.add_summary(summary_test_loss_str, i)
            #test_loss += loss_val
            
            if (i) % FLAGS.ae_lstm_save_dec_step == 0 or i == iteration-1: #20
                plot_traj_3d(dec_val,random_sequences,batch_seqlen,FLAGS.MAX_X,FLAGS.MAX_Y,i+1,FLAGS.ae_lstm_dec_dir)
            
            print("iter %d:" %(i+1))
            print("test loss: %f" %(np.sqrt(test_loss)*FLAGS.MAX_Y))#*pretrain_batch_size/num_test)*FLAGS.MAX_Y))
            #time.sleep(2)
          
            #loss_queue.append((train_loss/num_train, test_loss))
            duration = time.time() - start_time
            print("duration: %f s" %(duration))
            
        #save_path = saver.save(sess, model_ckpt)
        print("Model saved in file: %s" % save_path)#
        summary_writer.close()
        
def plot_traj_3d(Spred,Sgt,seqlen,MAX_X,MAX_Y,iteration,save_dir):
    num_player =5 
    img=mpimg.imread('raw/court.png')
    half_court = img[:,325:651,:]
    r, c = np.ogrid[0:half_court.shape[0], 0:half_court.shape[1]]
    #save_dir = 'decode'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for v in range(int(Sgt.shape[0]/num_player)):       
        t = np.arange(seqlen[v*num_player])
        hcplot = plt.imshow(half_court)
        plt.plot(Sgt[0+v*num_player,t,0]*MAX_X,Sgt[0+v*num_player,t,1]*MAX_Y,'r-', label='p1')
        plt.plot(Sgt[1+v*num_player,t,0]*MAX_X,Sgt[1+v*num_player,t,1]*MAX_Y,'g-', label='p2')
        plt.plot(Sgt[2+v*num_player,t,0]*MAX_X,Sgt[2+v*num_player,t,1]*MAX_Y,'b-', label='p3')
        plt.plot(Sgt[3+v*num_player,t,0]*MAX_X,Sgt[3+v*num_player,t,1]*MAX_Y,'y-', label='p4')
        plt.plot(Sgt[4+v*num_player,t,0]*MAX_X,Sgt[4+v*num_player,t,1]*MAX_Y,'c-', label='p5')

        plt.plot(Spred[0+v*num_player,t,0]*MAX_X,Spred[0+v*num_player,t,1]*MAX_Y,'r:')
        plt.plot(Spred[1+v*num_player,t,0]*MAX_X,Spred[1+v*num_player,t,1]*MAX_Y,'g:')
        plt.plot(Spred[2+v*num_player,t,0]*MAX_X,Spred[2+v*num_player,t,1]*MAX_Y,'b:')
        plt.plot(Spred[3+v*num_player,t,0]*MAX_X,Spred[3+v*num_player,t,1]*MAX_Y,'y:')
        plt.plot(Spred[4+v*num_player,t,0]*MAX_X,Spred[4+v*num_player,t,1]*MAX_Y,'c:')
        plt.legend()
        fig = plt.gcf()
        #plt.show(hcplot)        
        fig.savefig('{2}/v{0}_2d({1}).png'.format(v,iteration,save_dir), bbox_inches='tight')
        
        fig1 = plt.figure()
        #plt.subplot(122)
        ax1 = fig1.gca(projection='3d')
        ax1.plot_surface(r, c, 0, rstride=5, cstride=5, facecolors=half_court)
    

        " plot trajectories ground truth "
        ax1.plot(Sgt[0+v*num_player,t,1]*MAX_Y,Sgt[0+v*num_player,t,0]*MAX_X,t,'r-', label='p1')
        ax1.plot(Sgt[1+v*num_player,t,1]*MAX_Y,Sgt[1+v*num_player,t,0]*MAX_X,t,'g-', label='p2')
        ax1.plot(Sgt[2+v*num_player,t,1]*MAX_Y,Sgt[2+v*num_player,t,0]*MAX_X,t,'b-', label='p3')
        ax1.plot(Sgt[3+v*num_player,t,1]*MAX_Y,Sgt[3+v*num_player,t,0]*MAX_X,t,'y-', label='p4')
        ax1.plot(Sgt[4+v*num_player,t,1]*MAX_Y,Sgt[4+v*num_player,t,0]*MAX_X,t,'c-', label='p5')
    
        " plot trajectories prediction "
        ax1.plot(Spred[0+v*num_player,t,1]*MAX_Y,Spred[0+v*num_player,t,0]*MAX_X,t,'r:')
        ax1.plot(Spred[1+v*num_player,t,1]*MAX_Y,Spred[1+v*num_player,t,0]*MAX_X,t,'g:')
        ax1.plot(Spred[2+v*num_player,t,1]*MAX_Y,Spred[2+v*num_player,t,0]*MAX_X,t,'b:')
        ax1.plot(Spred[3+v*num_player,t,1]*MAX_Y,Spred[3+v*num_player,t,0]*MAX_X,t,'y:')
        ax1.plot(Spred[4+v*num_player,t,1]*MAX_Y,Spred[4+v*num_player,t,0]*MAX_X,t,'c:')    
        ax1.legend()
        ax1.autoscale(enable=True, axis='both', tight=True)
    
        #plt.show(fig1)
        pl.dump(fig1,open('{2}/v{0}_({1}).pickle'.format(v,iteration,save_dir),'wb'))
        fig1.savefig('{2}/v{0}_3d({1}).png'.format(v,iteration,save_dir), bbox_inches='tight')
        plt.close('all')