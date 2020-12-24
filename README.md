# PL-utils
> Utils for pytorch-lightning.


Collection of utils - datamodules, callbacks, models etc for pytorch-lighting.  

# Datamodules.

Imagenette.

Imagenette dataset Datamodule.  
Subset of ImageNet.  
https://github.com/fastai/imagenette  

```
from pl_utils.imagenette_datamodule import ImagenetteDataModule, ImageWoofDataModule
```

```
imagenette_datamodule = ImagenetteDataModule(data_dir=DATADIR)
```

```
imagenette_datamodule.setup()
```

```
len(imagenette_datamodule.train_dataset), len(imagenette_datamodule.val_dataset)
```




    (9469, 3925)



```
woof_datamodule = ImageWoofDataModule(data_dir=DATADIR)
```

```
woof_datamodule.setup()
```

```
len(woof_datamodule.train_dataset), len(woof_datamodule.val_dataset)
```




    (9025, 3929)


