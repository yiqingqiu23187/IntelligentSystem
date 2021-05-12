# 智能系统Lab1

By 黄子豪 18302010034

[TOC]



## 代码基本架构（java面向对象实现）

在这次lab中我主要使用Java面向对象设计了三个模拟神经网络的类，分别是Node模拟单个的神经元，Layer组织着同一层的神经元集合，Network进一步将Layer组织起来，形成bp网络，并且向外提供训练和测试的接口；在提交的代码中还有两个测试程序，分别是Sin和Classification，他们分别调用Network提供的抽象网络接口来训练和测试。

接下来自底向上地解释各个类的基本架构。

### Node：基本单位神经元

Node的实例是组成复杂网络的基本单位神经元，每个神经元记录着自己的输入、输出以及所有**输入的权重向量（包括bias）**，后来为了计算的方便，还增加了反向传播时需要迭代计算的delta，如下所示：

```java
public class Node {
    int weightSize;
    double[] inputWeights;
    double b;
    double input;
    double output;
    double delta;//并不是真正的delta w ,只是被求和项

    public Node(int weightSize) {
        this.weightSize = weightSize;
        inputWeights = new double[weightSize];

        for (int i = 0; i < weightSize; i++) {
            inputWeights[i] = Math.random()/1000;
        }
        b = -Math.random()/1000;
    }

    double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    double derivatives() {
        return output * (1 - output);
    }

    void forward() {
        output = sigmoid(input);
    }

}

```

节点中有一些简单的计算方法，包括由sigmoid计算输出、给出自己由于sigmoid求导带来的对梯度的影响项，请注意**我尝试过采用不同的激活函数，如tanh、Relu等，在这里讲述代码架构并不涉及**，后面讲述不同网络结构和参数对比时会详细展开。

### Layer：组织同一层的Node，统一正向计算和反向传播

每一个layer都管理着自己这一层的所有节点，并且有跟左边一层和右边一层的链接。当进行forward的时候，采用循环遍历所有节点，用这些节点的权重向量和左边一层的输出点乘，就得到了所有节点的输入，进一步调用节点本身的forward函数，设置好输出。当反向传播backward时，使用右边一层所有节点的delta和由于sigmoid带来的导数项（输出层除外），配合上本层的节点输出来设置权重的变化量。代码如下所示：

```java
public class Layer {
    Node[] nodes;
    int nodeNumber;
    Layer left;
    Layer right;

    //从左到右建立层
    public Layer(int nodeNumber, Layer left) {
        this.nodeNumber = nodeNumber;
        this.left = left;
        nodes = new Node[nodeNumber];
        for (int i = 0; i < nodeNumber; i++) {
            if (left != null) nodes[i] = new Node(left.nodeNumber);
            else nodes[i] = new Node(0);
        }
    }


    void forward() {
        for (Node thisNode : nodes) {
            double temp = 0;
            for (int i = 0; i < left.nodes.length; i++) {
                temp += left.nodes[i].output * thisNode.inputWeights[i];
            }
            temp += thisNode.b;

            thisNode.input = temp;
            thisNode.forward();

        }
    }


    void backward() {
        for (int i = 0; i < nodeNumber; i++) {
            double temp = 0;
            for (int j = 0; j < right.nodeNumber; j++) {
                Node node = right.nodes[j];
                temp += node.delta * node.derivatives() * node.inputWeights[i];
            }
            nodes[i].delta = temp;
        }
        for (int i = 0; i < nodeNumber; i++) {
            Node node = nodes[i];
            for (int j = 0; j < node.weightSize; j++) {
                node.inputWeights[j] += left.nodes[j].output * node.derivatives() * node.delta * Network.wLearningRate;// + -0.0005 + Math.random()/1000;
            }
            node.b += node.derivatives() * node.delta * Network.bLearningRate;//-0.0005 + Math.random()/1000;
        }

    }
}

```

现在我们已经能够设置每一层的输出，并且能够为每一层反向传播啦，接下来的任务就是把所有层组织称一个网络，有规律地从第一层到最后一层forward，从最后一层到第一层backward啦！

### Network：对内forward和backward，对外提供接口

层的工作已经完成，Network只需要接收我们传给它的参数，初始化层数和每一层的节点数，然后把它们连接起来，再循环往复的从输入层向输出层forward，从输出层向输入层backward。具体代码如下：

```java
public class Network {
    public static double wLearningRate = 0.005;
    public static double bLearningRate = 0.002;
    Layer[] layers;

    public Network(int[] nums) {
        layers = new Layer[nums.length];
        layers[0] = new Layer(nums[0], null);
        for (int i = 1; i < nums.length; i++) {
            layers[i] = new Layer(nums[i], layers[i - 1]);
        }
        for (int i = 0; i < nums.length - 1; i++) {
            layers[i].right = layers[i + 1];
        }
    }

    void trainSin(double[] inputs, double[] outputs) {
        forward(inputs);

        //最后一层的forward不一样
        Layer lastLayer = layers[layers.length - 1];
        for (Node thisNode : lastLayer.nodes) {
            thisNode.output = thisNode.input;
        }
        backward(outputs);
    }

    double testSin(double[] inputs, double[] outputs) {
        forward(inputs);

        //最后一层的forward不一样
        Layer lastLayer = layers[layers.length - 1];
        for (Node thisNode : lastLayer.nodes) {
            thisNode.output = thisNode.input;
        }

        return outputs[0] - layers[layers.length - 1].nodes[0].output;
    }

    void trainClassfi(double[] inputs, double[] outputs) {
        forward(inputs);

        softmax();

        backward(outputs);
    }


    boolean testClassfi(double[] inputs, double[] outputs) {
        forward(inputs);
        softmax();

        int index = -1;
        double max = 0;
        Layer lastLayer = layers[layers.length - 1];
        for (int i = 0; i < outputs.length; i++) {
            if (lastLayer.nodes[i].output > max) {
                index = i;
                max = lastLayer.nodes[i].output;
            }
        }
        if (outputs[index] == 1) return true;
        else return false;
    }

    int predictClassifi(double[] inputs) {
        forward(inputs);
        softmax();

        int index = -1;
        double max = 0;
        Layer lastLayer = layers[layers.length - 1];
        for (int i = 0; i < lastLayer.nodes.length; i++) {
            if (lastLayer.nodes[i].output > max) {
                index = i;
                max = lastLayer.nodes[i].output;
            }
        }
        return (index + 1);
    }

    void forward(double[] inputs) {
        for (int n = 0; n < inputs.length; n++) {
            layers[0].nodes[n].output = inputs[n];
        }

        for (int i = 1; i < layers.length; i++) {
            layers[i].forward();
        }
    }

    void backward(double[] outputs) {
        Layer lastLayer = layers[layers.length - 1];
        for (int i = 0; i < lastLayer.nodes.length; i++) {
            Node node = lastLayer.nodes[i];
            node.delta = outputs[i] - node.output;
            for (int j = 0; j < node.weightSize; j++) {
                node.inputWeights[j] += lastLayer.left.nodes[j].output * node.delta * Network.wLearningRate;
            }
            node.b += node.delta * Network.bLearningRate;
        }

        for (int i = layers.length - 2; i > 0; i--) {
            layers[i].backward();
        }
    }


    void softmax() {
        //最后一层的forward不一样
        Layer lastLayer = layers[layers.length - 1];
        double total = 0;
        for (Node thisNode : lastLayer.nodes) {
            total += Math.exp(thisNode.input);
        }
        for (Node thisNode : lastLayer.nodes) {
            thisNode.output = Math.exp(thisNode.input) / total;
        }
    }

}

```

值得一提的是，因为采用上述网络组织层，层组织节点的方式，**整个网络的层数、每一层的节点数都是灵活可变，易伸缩可调整的**，而调整的方式就是改变传入Network构造函数的数组，例如拟合sin函数时传入的是new int[]{1,50,1}，手写汉字分类时传入的是new int[]{28*28,64,12}；而学习率也在Network层中可调节，这满足了lab的第一点要求，即可灵活调整的参数。

### Sin/Classification：实例化network，进行训练并测试

Network层提供了初始化接口Network()和train以及test函数，Sin和Classification根据各自任务的不同复用这个网络，初始化训练集和测试集，并且进行训练和测试，代码如下：

```java
public class Sin {
    public static void main(String[] args) {
        int sampleNum = 400;
        double[][] input = new double[sampleNum][1];
        double[][] output = new double[sampleNum][1];
        for (int i = 0; i < sampleNum; i++) {
            input[i][0] = Math.PI * 2 * i / sampleNum - Math.PI;
            output[i][0] = Math.sin(input[i][0]);
        }

        int testSize = 400;
        double[][] test_inputs = new double[testSize][1];
        double[][] test_outputs = new double[testSize][1];
        for (int i = 0; i < testSize; i++) {
            test_inputs[i][0] = Math.PI * 2 * Math.random() - Math.PI;
            test_outputs[i][0] = Math.sin(test_inputs[i][0]);
        }
        Network bp = new Network(new int[]{1, 50, 1});
        for (int j = 0; j < 100000; j++) {
            for (int i = 0; i < sampleNum; i++) {
                bp.trainSin(input[i], output[i]);
            }
        }

        double totalError = 0;

        for (int i = 0; i < testSize; i++) {
            totalError += Math.pow(bp.testSin(test_inputs[i], test_outputs[i]), 2);
        }
        System.out.println("totalerror for test set:" + totalError / testSize);


    }
}

```

```
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Classification {

    public static void main(String[] args) {
        int trainSize = 450;
        int testSize = 620 - trainSize;
        double[][][] input = new double[trainSize][12][28 * 28];//这样的顺序是为了提高局部性
        double[][] output = new double[12][12];
        for (int i = 0; i < trainSize; i++) {
            for (int j = 0; j < 12; j++) {
                input[i][j] = imgInfo("train/" + (j + 1) + "/" + (i + 1) + ".bmp");
            }
        }

        double[][][] test_input = new double[testSize][12][28 * 28];//这样的顺序是为了提高局部性
        for (int i = 0; i < testSize; i++) {
            for (int j = 0; j < 12; j++) {
                test_input[i][j] = imgInfo("train/" + (j + 1) + "/" + (i + 1 + trainSize) + ".bmp");
            }
        }
        for (int i = 0; i < 12; i++) {
            output[i][i] = 1;
        }

        Network network = new Network(new int[]{28 * 28, 64, 12});
        int totalEp = 0;
        double lastRate = -1;
        double rate = 0;
        int ep = 10;
        while (totalEp < 60) {
            lastRate = rate;
            totalEp += ep;
            for (int i = 0; i < ep; i++) {
                for (int j = 0; j < trainSize; j++) {
                    for (int k = 0; k < 12; k++) {
                        network.trainClassfi(input[j][k], output[k]);
                    }
                }
            }
//            int right = 0;
//            for (int j = 0; j < testSize; j++) {
//                for (int k = 0; k < 12; k++) {
//                    if (network.testClassfi(test_input[j][k], output[k])) right++;
//                }
//            }
//            rate = right / (testSize * 12.0);
        }

        System.out.println();
        for (int i = 0; i < 12; i++) {
            System.out.println(network.predictClassifi(test_input[50][i]));
        }

    }


    public static double[] imgInfo(String src) {
        // 读取图片到BufferedImage
        BufferedImage bf = readImage(src);//绝对路径+文件名
        // 将图片转换为二维数组
        int[] rgbArray1 = convertImageToArray(bf);
        double[] imgInfo = new double[rgbArray1.length];
        for (int i = 0; i < rgbArray1.length; i++) {
            if (rgbArray1[i] == -1) {
                imgInfo[i] = 0;
            } else {
                imgInfo[i] = 1;
            }
        }

        return imgInfo;
    }

    public static BufferedImage readImage(String imageFile) {
        File file = new File(imageFile);
        BufferedImage bf = null;
        try {
            bf = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bf;
    }

    public static int[] convertImageToArray(BufferedImage bf) {
        // 获取图片宽度和高度
        int width = bf.getWidth();
        int height = bf.getHeight();
        // 将图片sRGB数据写入一维数组
        int[] data = new int[width * height];
        bf.getRGB(0, 0, width, height, data, 0, width);
        return data;
    }
}

```

必须注意的是上述代码均是为了展现基本架构，所有优化和尝试并没有展现。初始版本图片分类任务中正确率稳定在83%左右，后续通过参数调整、数据增加等手段提升到了89%左右。



## 实验对比和优化分析

### 不同网络结构、网络参数的实验比较

#### 层数：

> 一般认为，增加隐层数可以提高网络复杂度，提高拟合能力，从而可以降低网络误差或者是提高精度，但是也增加了网络的训练时间和出现“过拟合”的可能性，并且也使得调整网络每层的节点数和其他参数变得困难。

而本次lab的两个任务，拟合正弦函数和十二分类问题，都不是非常复杂，因此本次lab实验中我只通过分类图片的准确率对比三层网络（一个隐层）和四层网络（两个隐层）的区别。

![image-20201101190945039](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20201101190945039.png)

可以看到，对于同一分类任务同一训练集和测试集，三层网络的表现要远好于四层网络，所需节点数和复杂度也更少、更低。在粗略的测试中三层网络大概在60-70个节点左右准确率达到最大（83%左右），而两个采取了三层网络中最优解的四层网络（第二个隐层分别为50个节点和70个节点），不仅需要更多的第一层隐层节点，准确率也更低，不到50%。

这样的结果可能是由于层数过高模型复杂度过大，但实际样本集和任务都不需要这么大的复杂度从而导致模型过拟合，因此本次lab中我最终采用3层的模型来拟合正弦函数和分类手写汉字。

#### 隐层节点数：

在对于恰当的层数的实验比较中已经初步确定对于一个三层的网络，其最佳隐层节点数应该在55-70范围内，因此在这样的预实验基础上接下来只需对这个范围内的节点数进行检查，此外由于这一区间范围很小，因此为了避免偶然性，以下图表的每个数据都是测量了两次取其平均值：

![image-20201101193242835](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20201101193242835.png)

在60-70节点数范围内，测得的准确率的平均值分别是80.7%，83.3%，83.7%，83.4%，83.3和82.8%，可以看出在这个范围内准确率其实已经变化不大，我结合输入层为m=28*28=784个节点以及输出层为l=12个节点，最终取隐层节点数为64（https://www.zybuluo.com/hanbingtao/note/476663中建议隐层节点数设置为ml之积的平方根的倍数）。

#### 学习率：

由于在Back Propagation权重的梯度方向比bias的梯度方向要多一项前继节点的输出，因此为了使得普通节点的权重变化幅度和bias变化幅度大致相同，必须对于权重采用更大的学习率而bias采用更小的学习率，这在我的代码中也有体现，分别是Network.wlearningrate和Network.blearningrate，在尝试调节这两个学习率的过程中我采用了多个组合，并且结合了一些参考文献确定620张图片的分类任务学习率应该在0.001的量级，各个组合对应的准确率如下：

| wlearningrate/blearningrate | 拟合sin误差 | 图片分类准确率 |
| --------------------------- | ----------- | -------------- |
| 0.01/0.005                  | 0.0082      | 78%            |
| 0.008/0.005                 | 0.0031      | 83%            |
| 0.005/0.002                 | 0.0012      | 88%            |
| 0.003/0.001                 | 0.0011      | 85%            |

对于拟合正弦函数，学习率组合为（0.005,0.002）和（0.003,0.001）都比较好，而对于图片分类来讲（0.005,0.002）效果明显更好，因此最后选择了第三组。

#### 初始权重范围：

提到这一点必须感谢汪励颢助教，由于使用java实现整个网络结构，权重初始值我直接采用的Math.random()，即（0,1）的范围，bias则是（-1,0）的范围，事实上对于正弦函数的拟合这样的函数拟合效果很好；但是对于更为复杂的问题，如图片分类，由于输入向量节点变多网络变复杂，必须使用更小的初始值，否则会花费很久的时间才能达到收敛。所以在把初始权重和bias缩小1000倍以后训练网络的时间终于缩小到可以接受的10分钟，准确率也到了88.3%。

#### 拟合和分类结果

采用上述参数对比中的最优解，我的图片分类问题最终达到了88.7%的正确率，我的正弦拟合图像如下：

拟合结果：

![Figure_3](C:\Users\LENOVO\Desktop\Figure_3.png)

测试集结果：

![Figure_1](C:\Users\LENOVO\Desktop\Figure_1.png)

损失函数：

![Figure_2](C:\Users\LENOVO\Desktop\Figure_2.png)

### 实验优化和改进

#### 正则项

其实由于上述参数调整的过程中，我的网络结构已经趋于饱和，并未发生明显过拟合的情况，因此添加正则项并没有减小拟合的误差或者是提高分类准确率：

|              | 拟合sin误差 | 图片分类准确率 |
| ------------ | ----------- | -------------- |
| **有正则项** | 0.0022      | 87.5%          |
| **无正则项** | 0.0012      | 88.3%          |

#### 不同的激活函数

sigmoid函数虽然有很多很好的特性，但是有一个致命的弱点，就是在曲线两端存在着梯度消失的情况，因此我尝试过不同的激活函数。

Tanh:
$$
Tanh(x)=(exp(x)-exp(-x))/(exp(x)+(exp(-x))
$$
ReLU:
$$
ReLU(x)=max(0,x)
$$
cos:
$$
cos(x)=cosx
$$
ELU:
$$
ELU(x)=\left\{
\begin{aligned}
x,if &&x>0  \\
\alpha (exp(x)-1),if &&x<0\\
\end{aligned}
\right.
$$
然后可能是由于问题并不复杂，模型也是简单的三层结构，因此使用上述公式并没有给我带来多大的收益，使用ReLU甚至会使我的模型爆掉，最后还是采用的sigmoid函数。



## 对反向传播算法的理解——犯错和校正的过程

### *由高中的线性回归开始*

其实当我真的手撕了一个bp网络之后，回过头来才发现这种思想早已有了来源，高中时期数学上学过而物理上经常要用到的线性回归可以算是反向传播算法的一种体现，即给定一组输入x向量和期望输出t向量，试图找到一个合适的函数f，使得
$$
Min(t-f(x))
$$
在线性回归中，这样的函数f其实已经确定为一次线性函数，我们要做的只不过是确定它的斜率和截距两个参数，因此通过理性数学推导的方法完全可以计算得出二者的公式。

但现在情况不同，我们试图用神经网络模型拟合并预测一切可能的函数。

> In the mathematical theory of artificial neural networks, the **[universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)** states that a feed-forward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of **R**, under mild assumptions on the activation function.

*用更易懂的话说，那就是：我们尝试成为上帝或者至少说是先知，预测一切。*

*——用简单的暴力计算的方式。*

曾经经典物理认为已经物体的初始运动状态和受力情况，有能力预测该物体在未来任意时刻的位置和运动矢量，但是量子力学带来了不确定性，指出在微观的尺度上这种不确定性是不可预知的——而现在我们似乎回到了起点，只不过这一次我们用的不再是经典物理那么完美逻辑的工具，而是更为强大的计算能力和导数。

### *由线性到非线性：神经网络的登陆*

好啦回到这个神经网络模型上来，它的名字来源于单个节点的行为确实类似于生物的神经元，而它的行为也确实像一个真实的人：犯错误，尝试让错误最小化，改变自己的参数，重新测试。

但是现在遇到了一个问题，线性模型不能解决线性不可分的问题，就像我一开始不停地调节学习率并不能解决隐层节点过少的问题，很自然的，需要引入非线性模型。

虽然这次的实验我尝试过不同的非线性函数比如tanh、relu、elu，但是其实效果都不如sigmoid函数（也许这本来就是sigmoid被广泛采用的原因吧），sigmoid在我看来有惊奇的对称美：
$$
sigmoid(x)=1/(1+exp(-x))
$$
她是以（0，0.5）中心对称的，在中央变化较大，在两端则和蔼地包容（当然这也带来了梯度消失的坏处，花了我好长的时候调整learningrate才解决）。

### *反向传播算法的精髓：由误差更新权重*

权重的参数是在一个范围内随机初始的，由这个初始值计算出输出，并与期望输出进行计算得出误差，再由梯度下降法和链式求导法则更新权重，这是我理解的反向传播算法的精髓。通过如此循环往复的“犯错、最小化误差、更新参数”的过程，反向传播算法不仅告诉我们当我们改变weight和bias的时候，损失函数改变的速度，反向传播也告诉我们如何改变权值和偏置以改变神经网络的整体表现。

### *写在最后：个人的一点想法*

一直以来我都崇尚理论，尤其喜欢数学和物理，那种推导令人信服，公式的美似乎隐藏着某种哲学，我相信建立在这样的理论体系之上的人类文明的发展是可靠的，事实也正是如此。但是如今的”人工智能“大火，似乎已经被各国和各大公司视为人类的未来，甚至大有人工智能取代真人的说法；然后在我初步了解它背后的原理之后，怎么看来都只是蛮力——最多是结合了技巧的蛮力，其本质还是在不断地试错然后修正，有时候模型复杂到创建者都无法理解，那么它的发展我们又怎么得到保障？这样看来至于人工智能取代人类的说法，只不过是外行人的天马行空罢了。

但是不论如何我还是希望以后这个领域能发展的越来越好，能有更多的像人脸识别这样的成功应用，当然如果能发展出更有理论支持的科学就更好了！