# Automated Part Mix-n-Match of Chair 3D Models
This system randomly mixes and deforms the parts of given chair models from the PartNet-SymH dataset, producing a brand new and unique chair, which is then evaluated to show how “good” or “bad” the chair is. The chairs are mixed and based on necessity deformed using Coherent Point Drift algorithm, before being sequentially attached back together. The new chair is then evaluated using a LeNet based scorer, a ResNet34 based scorer and a PointNet based scorer.

# Modules
## Parser
The parser runs on the PartNet-Symh dataset obb files to create a dataset of Mesh objects from the set of models provided to the parser. Each Mesh object contains a list of Part objects. Each Part object corresponds to a physical part of the object (found in the .obj file).

## Mixer
The mixer chooses a target Mesh object at random from the dataset created by the parser. Each chair part in the target is to be replaced by corresponding chair parts sourced from other chairs (chosen at random) in the dataset. The target mesh only provides the base structure on which the replacement and deformation is based. 

Each chair part in the target is either -
- Replaced by parts deformed to match the shape of the target parts
- Directly replaced by the source parts
- A transformed target part, in case no match is found in the dataset (example an armrest)

To implement CPD for part deformation we used the library Probreg (probablistic point cloud registration library), which uses Open3D as an interface and implements various kinds of point cloud registration algorithms, of both rigid and non rigid kind. 


![alt text](https://github.com/Atmika293/gm-project/blob/master/full_result.png)
A pictorial represntation of mixing methodology

## Plausibility Scorer
For the plausibility scorer, we trained 3 different neural networks:
- LeNet
- ResNet
- PointNet
LeNet is unable to clearly distinguish between a plausible chair and chairs with missing or unusually deformed parts (“bad” chairs). Both types of chairs are given high plausibility scores.
ResNet scores “bad” chairs low, but does not give a much higher score to plausible chairs.
PointNet can make a better distinction in plausibility of the generated chairs. Plausible chairs are given a high score, while “bad” chairs are given low scores. 

# References
- Probreg
@software{probreg,
    author = {{Kenta-Tanaka et al.}},
    title = {probreg},
    url = {https://probreg.readthedocs.io/en/latest/},
    version = {0.1.6},
    date = {2019-9-29},
}
- PartNet-SymH
@InProceedings{Yu_2019_CVPR,
    title = {{PartNet}: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation},
    author = {Fenggen Yu and Kun Liu and Yan Zhang and Chenyang Zhu and Kai Xu},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {to appear},
    month = {June},
    year = {2019}
}
@InProceedings{Mo_2019_CVPR,
    author = {Mo, Kaichun and Zhu, Shilin and Chang, Angel X. and Yi, Li and Tripathi, Subarna and Guibas, Leonidas J. and Su, Hao},
    title = {{PartNet}: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level {3D} Object Understanding},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
