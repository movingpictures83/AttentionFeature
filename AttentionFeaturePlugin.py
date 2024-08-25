import PyPluMA
import PyIO
import pickle
from tqdm import tqdm
import torch

def output_feature_attention(attn):
    spatial_attn, feature_attn = attn
    att_mat = feature_attn
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1).cpu()
    #att_mat = att_mat[-1]

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    final_attn = v[0, 1:]/v[0, 1:].sum()
    return final_attn


class AttentionFeaturePlugin:

  def input(self, inputfile):
     self.parameters = PyIO.readParameters(inputfile)

  def run(self):
      pass

  def output(self, outputfile):
   features = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["features"])

   pickle1 = open(PyPluMA.prefix()+"/"+self.parameters["attn"], "rb")
   all_attn = pickle.load(pickle1)
   pickle2 = open(PyPluMA.prefix()+"/"+self.parameters["attn_native"], "rb")
   all_attn_native = pickle.load(pickle2)
   
   attn_dict = {'attn':[], 'feature':[], 'label':[], 'pred_label':[], 'PPI':[], 'misclassified':[]}
   predicted_labels = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["predict"])
   ppi_list = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["ppi"])
   labels = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["labels"])
   predicted_labels_native = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["nativepredict"])
   native_ppi_list = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["nativeppi"])
   

   for i, attn_i in tqdm(enumerate(all_attn)):
    feature_att = output_feature_attention(attn_i)
    for j in range(len(features)):
     attn_dict['attn'].append(float(feature_att[j].cpu()))
     attn_dict['feature'].append(features[j])

    for j in range(5):
        attn_dict['label'].append(labels[i])
        attn_dict['pred_label'].append(predicted_labels[i])
        attn_dict['PPI'].append(ppi_list[i])
        attn_dict['misclassified'].append(labels[i]!=predicted_labels[i])


   for i, attn_i in tqdm(enumerate(all_attn_native)):
    feature_att = output_feature_attention(attn_i)
    for j in range(len(features)):
     attn_dict['attn'].append(float(feature_att[j].cpu()))
     attn_dict['feature'].append(features[j])

    for j in range(5):
        attn_dict['label'].append(1)
        attn_dict['pred_label'].append(predicted_labels_native[i])
        attn_dict['PPI'].append(native_ppi_list[i])
        attn_dict['misclassified'].append(predicted_labels_native[i]!=predicted_labels_native[i])

   
   import pandas as pd
   attn_df_test = pd.DataFrame(attn_dict)
   myout = open(outputfile, "wb")
   pickle.dump(attn_df_test, myout)

