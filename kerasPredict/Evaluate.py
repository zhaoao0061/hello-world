import numpy as np

def linear_predict(model_linear):
   pos_x = []
   pos_target = []
   #pos_target = tpc.toTen(pos_target, 101)
   pos_x_cnn = np.reshape(pos_x, [pos_x.shape[0], 1, 100, 5])
   #pos_x_test,y_train = tpc.xTo2D(X_train,y_train,150)
   predict_ten = model_linear.predict([pos_x,pos_x_cnn])
   #believe_P = np.max(predict_ten,1) * 100
   predict_ten *= 100
   pos_target *= 100
   predict_ten = np.clip(predict_ten,0,100)
   pos_target = np.reshape(np.clip(pos_target,0,100),[len(pos_target),1])
   return predict_ten,pos_target


def WMAE(pred,true):
   # yibai = np.array(range(0,101))
   # yibai_2 = (np.array(yibai)/100-0.5)*4
   # c = 0
   # ly = np.clip((1 + yibai_2 + c) / (1 - yibai_2 + c),1e-2,100)
   # wy = np.log(ly)
   #wy = yibai_2 ** 3
   # plt.plot(wy)
   pred = np.clip(np.array(pred)/50-1, -0.99,0.99)
   true = np.clip(np.array(true)/50-1, -0.99,0.99)
   pred = pred[:len(true)]
   c=0.001

   Pred = 0.5 * np.log((1 + pred) / (1 - pred))
   Target = 0.5* np.log((1 + true) / (1- true))
   Target = np.reshape(Target,[len(Target),1])
   wab = np.abs(Pred-Target)
   wsq = np.square(Pred-Target)
   wmae = np.sum(wab,keepdims=False)/len(wab)
   return wmae

def Recall(pred,target,below = 5):  # 查全误差
   sum_dif = 0
   n = 0.00000001
   dif = 0
   for i in range(len(target)):
      if target[i] <= below:
         dif = pred[i] - target[i]
         sum_dif += dif
         n += 1
   mean_error = sum_dif / n
   print('Recall:', mean_error)
   if n >= 1:
      return mean_error
   else:
      return [-1]


# 查准误差
#预测值小于x时，与真实值的误差
def Turn_error(pred,target,below = 5):
   sum_dif = 0
   n = 0.00000001
   for i in range(1,len(target)):
      if pred[i] <= below and pred[i-1] > below:
         print(pred[i - 1])
         print(pred[i])
         dif = np.abs(target[i] - pred[i])
         sum_dif += dif
         n += 1
   mean_error = sum_dif / n
   print('Turn_error:', mean_error)
   if n >= 1:
      return mean_error
   else:
      return [-1]

# 查准误差
#预测值小于x时，与真实值的误差
def up_Turn_error(pred,target,below = 5):
   sum_dif = 0
   n = 0.00000001
   for i in range(1,len(target)):
      if pred[i] >= below and pred[i-1] < below:
         print(pred[i - 1])
         print(pred[i])
         dif = np.abs(target[i] - pred[i])
         sum_dif += dif
         n += 1
   mean_error = sum_dif / n
   print('up_Turn_error:', mean_error)
   if n >= 1:
      return mean_error
   else:
      return [-1]


def TAC(pred, target):
   right = 0
   sum_pred_turn = 0
   for i in range(len(target)):
      if pred[i] == 0:
         if target[i] == 0:
            right += 1
         sum_pred_turn += 1
   if sum_pred_turn > 0:
      return right/sum_pred_turn*100
   else:
      return -1

def save_result(save_text, model_path,index, epoch ):
   eval_save = open(model_path + 'result.txt', 'a')
   eval_save.write('index: ' + str(index) + ' epoch:'+ str(epoch) + '\n')
   # eval_result = [['tac','wmae','TE','Recall']]
   eval_save.write(str(save_text) + '\n')

# def save_evaluate(model_linear,model_path,pos_range,test_codesh,below = 5):
#  eval_save = open(model_path + 'result.txt', 'a')
#  eval_save.write('index: ' + str(index) + ' epoch:'+ str(epoch) + '\n')
#  eval_result = [['code','tac','wmae','TE','Recall']]
#  recall_l = []
#  tac_l = []
#  wmae_l = []
#  te_l = []
#
#  for code in test_codes:
#     pred, pos_target,pos_cls = linear_predict(model_linear, pos_range, code=code)
#     recall = Recall(pred, pos_target, below=below)
#     tac = TAC(pred, pos_target)
#     wmae = WMAE(pred, pos_target)
#     te = Turn_error(pred, pos_target,below = below)
#     temeval = {'code':code,'Turn Error':te[0],'recall:':recall[0],'tac':tac, 'wmae':wmae}
#     eval_save.write(str(temeval) + '\n')
#     eval_result.append([code,tac,wmae,te,recall])
#     temeval = {}
#     recall_l.append(recall[0])
#     te_l.append(te[0])
#     wmae_l.append(wmae)
#     tac_l.append(tac)
#  #eval_save.write('eval_result: ' + str(eval_result) + '\n')
#  while -1 in recall_l:
#     recall_l.remove(-1)
#  while -1 in te_l:
#     te_l.remove(-1)
#  ave_recall = np.average(recall_l)
#  ave_turn_error = np.average(te_l)
#  eval_save.write('ave_recall:'+ str(ave_recall) + '  ave_turn error:' + str(ave_turn_error)+'\n')
#  eval_save.close()
#  return eval_result,[recall_l,te_l,wmae_l,tac_l,ave_recall,ave_turn_error]


def evalute_result(pred,true):
   sum_div = 0
   n = -0.00000001
   div=0
   for i in range(len(true)):
      if pred[i] == 0:
         div = true[i] - pred[i]
         sum_div += div
         n += 1
   mean_error = sum_div / n
   summean = (100-mean_error)*n / 10
   print('mean_error:',mean_error,summean)

   return mean_error,summean

