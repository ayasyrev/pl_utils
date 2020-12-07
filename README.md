# PL-utils
> Utils for pytorch-lightning.


Collection of utils - datamodules, callbacks, models etc for pytorch-lighting.  

# Datamodules.

Imagenette.

Imagenette dataset Datamodule.  
Subset of ImageNet.  
https://github.com/fastai/imagenette  

```python
from pl_utils.imagenette_datamodule import ImagenetteDataModule, ImageWoofDataModule
```

```python
imagenette_datamodule = ImagenetteDataModule(data_dir=DATADIR)
```

```python
imagenette_datamodule.setup()
```

```python
len(imagenette_datamodule.train_dataset), len(imagenette_datamodule.val_dataset)
```




    (9469, 3925)



```python
woof_datamodule = ImageWoofDataModule(data_dir=DATADIR)
```

```python
woof_datamodule.setup()
```

```python
len(woof_datamodule.train_dataset), len(woof_datamodule.val_dataset)
```




    (9025, 3929)


