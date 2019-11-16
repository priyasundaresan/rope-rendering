import os

def gen_dir(name, num_vids):
    if os.path.exists("./%s" % name):
        os.system('rm -rf ./%s' % name)
    os.makedirs('./%s' % name)
    for i in range(num_vids):
        os.system('/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app/Contents/MacOS/blender -b -P rope-blender.py')
        folder_name = 'vid%04d' % i
        os.system('mv images ./%s/%s/' % (name, folder_name))

if __name__ == '__main__':
    gen_dir('rope_dset', 3)
