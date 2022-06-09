import os
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks.tensorboard import *
from fastai.vision.gan import *
from deoldify.generators import *
from deoldify.critics import *
from deoldify.dataset import *
from deoldify.loss import *
from deoldify.save import *
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile




path = Path('data/')
path_hr = path
path_lr = path/'bandw'

proj_id = 'StableModel'

gen_name = proj_id + '_gen'
pre_gen_name = gen_name + '_0'
crit_name = proj_id + '_crit'

name_gen = proj_id + '_image_gen'
path_gen = path/name_gen

TENSORBOARD_PATH = Path('data/tensorboard/' + proj_id)

nf_factor = 2
pct_start = 1e-8


def get_data(bs: int, sz: int, keep_pct: float):
    return get_colorize_data(sz=sz, bs=bs, crappy_path=path_lr, good_path=path_hr,
                             random_seed=None, keep_pct=keep_pct)


def get_crit_data(classes, bs, sz):
    src = ImageList.from_folder(path, include=classes, recurse=True).split_by_rand_pct(0.1, seed=42)

    ll = src.label_from_folder(classes=classes)

    data = (ll.transform(get_transforms(max_zoom=2.), size=sz)
            .databunch(bs=bs).normalize(imagenet_stats))

    return data


def create_training_images(fn, i):
    dest = path_lr / fn.relative_to(path_hr)

    dest.parent.mkdir(parents=True, exist_ok=True)

    img = PIL.Image.open(fn).convert('LA').convert('RGB')

    img.save(dest)


def save_preds(dl):
    i = 0
    names = dl.dataset.items

    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen / names[i].name)
            i += 1


def save_gen_images(keep_pct=0.085):
    if path_gen.exists(): shutil.rmtree(path_gen)
    path_gen.mkdir(exist_ok=True)

    data_gen = get_data(bs=bs, sz=sz, keep_pct=keep_pct)

    save_preds(data_gen.fix_dl)

    PIL.Image.open(path_gen.ls()[0])



if not path_lr.exists():
    il = ImageList.from_folder(path_hr)
    parallel(create_training_images, il.items)

pretrain_learner_path = 'ColorizeStable_gen'
pretrain_critic_path = 'ColorizeStable_crit'


learn_crit=None
learn_gen=None
gc.collect()


bs = 1
sz = 192
keep_pct = 1.0
lr=2e-5


data_gen = get_data(bs=bs, sz=sz, keep_pct=keep_pct)
learn_gen = gen_learner_wide(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor)

learn_gen.load(pretrain_learner_path, with_opt=False)

save_gen_images(1.00)

data_crit = get_crit_data([name_gen, 'test'], bs=bs, sz=sz)

learn_crit = colorize_crit_learner(data=data_crit, nf=256).load(pretrain_critic_path , with_opt=False)

learn_gen.freeze_to(-1)
learn_crit.freeze_to(-1)

switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.0, 1.5), show_img=False, switcher=switcher,

                                 opt_func=partial(optim.Adam, betas=(0., 0.9)), wd=1e-3)

learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

learn.callback_fns.append(partial(GANTensorboardWriter, base_dir=TENSORBOARD_PATH, name='GanLearner', visual_iters=100))

learn.callback_fns.append(partial(GANSaveCallback, learn_gen=learn_gen, filename=pre_gen_name, save_iters=100))


learn.data = get_data(sz=sz, bs=bs, keep_pct=keep_pct)

learn.fit(1,lr)

learn.save('Model_final')