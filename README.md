# TensorFlow Models

Barvins fork of tensorflow models.
Changes done to deeplab model:
* Allow to train multiple classes per pixel, makes it suitable not only for segmentation but also for keypoint detection.
* Custom random data augmentation:
    * removed zoom since tensorflow's downsalmpling distorts fine details in images like nets. It was easier to provide different zoom levels in generated dataset than to implement proper resizing method in tensorflow.
    * random gamma
    * random contrast/brightness that does not go outside of min/max pixel value range
    * random light color
    * random hue in really wide range, we don't care about blue human faces, but want t-shirts to be in every possible color
    * random blur
    * random crop that assumes that dataset is generated so that there is always enough margin in image to not cut off parts of important objects
* pre and postprocessing layers can be cut off when exporting models, to get rid of operations not supported by TensorRT and Tensorflow Lite
`* ResizeBilinear layers are replaced with ResizeNearestneighbor, since there is no real difference in quality and nearst neighbor is easier to implement in CUDA, since TensorRT does not support resize layers.


This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)
