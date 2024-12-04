import os
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

def create_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def gen_aug_states(obs, env_name):
    if('Circle' in env_name):
        return obs[-16:]
    elif('Goal' in env_name):
        return obs[-16*3:]
    else:
        raise NotImplementedError

def stringify(values, lst, start, end):
    for v, i in zip(values, range(start, end)):
        x = str(v)
        x = str(x).strip('[]')
        #remove all ',' and ' ' in x
        x = re.sub(r'[,\s]', '', x)
        lst[i] = x
        

def compute_train_stats(
        bin_is_good_pred: torch.Tensor,
        labels_bad_is_one: torch.Tensor,
    ):
        with torch.no_grad():
            # Logits of the discriminator output; >0 for bad samples, <0 for good.
            # bin_is_good_pred = disc_logits_bad_is_high < 0
            # Binary label, so 1 is for bad, 0 is for good.
            
            bin_is_good_true = labels_bad_is_one == 0
            bin_is_bad_true = torch.logical_not(bin_is_good_true)
            bin_is_bad_pred = torch.logical_not(bin_is_good_pred)
         
            int_is_good_pred = bin_is_good_pred.long()
            int_is_good_true = bin_is_good_true.long()
            n_good = float(torch.sum(int_is_good_true))
            n_labels = float(len(labels_bad_is_one))
            n_bad = n_labels - n_good
            pct_bad = n_bad / float(n_labels) if n_labels > 0 else float("NaN")
            n_bad_pred = int(n_labels - torch.sum(int_is_good_pred))
            if n_labels > 0:
                pct_bad_pred = n_bad_pred / float(n_labels)
            else:
                pct_bad_pred = float("NaN")
            correct_vec = torch.eq(bin_is_good_pred, bin_is_good_true)
            acc = torch.mean(correct_vec.float())

            _n_pred_bad = torch.sum(torch.logical_and(bin_is_bad_true, correct_vec))
            if n_bad < 1:
                bad_acc = float("NaN")
            else:
                # float() is defensive, since we cannot divide Torch tensors by
                # Python ints
                bad_acc = _n_pred_bad.item() / float(n_bad)

            _n_pred_gen = torch.sum(torch.logical_and(bin_is_good_true, correct_vec))
            _n_gen_or_1 = max(1, n_good)
            good_acc = _n_pred_gen / float(_n_gen_or_1)

        return {
            "disc_acc": float(acc),
            "disc_acc_bad": float(bad_acc),  # accuracy on just bad examples
            "disc_acc_good": float(good_acc),  # accuracy on just good examples
        }

def MixUp(self, batch, labels, mini_batch_size, device):
        #MixUp
        good_batch = batch[:mini_batch_size]
        bad_batch = batch[mini_batch_size:]
        good_labels = labels[:mini_batch_size]
        bad_labels = labels[mini_batch_size:]
        
        lam = np.random.beta(0.2, 0.2, size=(len(good_batch), 1))
        lam = torch.from_numpy(lam).float().to(device)
        mixed_batch = lam*good_batch + (1-lam)*bad_batch
        mixed_labels = lam*good_labels + (1-lam)*bad_labels
        
        return mixed_batch, mixed_labels

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', costs=0, clfs_costs=0, rewards=0):
        #Mess with this to change frame size
        plt.figure(figsize=(7, 4), dpi=100)

        if(frames[0].shape[0] == 3):
            frames = [frame.transpose(1,2,0) for frame in frames]
            patch = plt.imshow(frames[0])
        elif(frames[0].shape[0] == 9):
            frames = [frame.reshape(3,3,64,320)[-1].transpose(1,2,0) for frame in frames]
            patch = plt.imshow(frames[0])
        elif(len(frames[0].shape) == 2):
            patch = plt.imshow(frames[0], cmap='gray')
        else:
            patch = plt.imshow(frames[0])
        plt.title(f'Cost: {costs[0]:.2f} | CLFS Cost: {clfs_costs[0]:.2f} | Reward: {rewards[0]:.2f}')
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])
            plt.title(f'Cost: {costs[i]:.2f} | CLFS Cost: {clfs_costs[i]:.2f} | Reward: {rewards[i]:.2f}')
            
        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(path + filename, fps=10)
        plt.close()

def hinge_loss(logits, labels):
    
    loss = torch.zeros_like(labels)
    loss[labels<1] = torch.nn.functional.softplus(logits[labels<1])  # Penalize positive predictions when the true label is negative

    return loss.mean()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def batched_inf(net, x, batch_size):
    with torch.no_grad():
        n = x.shape[0]
        y = []
        xs = x.split(batch_size)
        for x in xs:
            y.append(net(x.to('cuda')).cpu())
        y = torch.cat(y, dim=0)
        # def func(x):
        #     x = x.to('cuda')
        #     return net(x).detach()
        
        # y = torch.vmap(func)(x)

        assert y.shape[0] == n
        return y
        
