import os
import shutil


if __name__ == "__main__":
    sourcepath = 'framework_logs/cat/8_40_1.0/'
    target_path = 'tmp/cat'
    for cls in os.listdir(sourcepath):
        if cls[0] != '1':
            continue
        out_name = cls
        source_cls = os.path.join(sourcepath, cls) + '/checkpoints'
        target_cls = os.path.join(target_path, cls)+ '/checkpoints'
        save_name = os.path.join(target_path, out_name)
        
        if os.path.exists(source_cls):
            if not os.path.exists(target_cls):
                shutil.copytree(source_cls, target_cls) 
                #os.rename(target_cls, save_name)