import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    """
    Converts FPN's predictions: box coordinate offsets (dx,dy,dw,dh), box confidence and class probablities (both are logits, and hence in range [-1,1]) 
    into
    image-scale final box coordinates (x,y,w,h), box confidence and class probabilities (both mapped to range [0,1])

    x = sigmoid(dx) + cx
    y = sigmoid(dy) + cy
        cx and cy -> x and y coordinates of the top-left corner of the grid cell containing the center of the bounding box.
        dx and dy -> offsets from the location of the cell where the object was detected, and they're usually passed through a sigmoid function to ensure their values are between 0 and 1.
    w = exp(dw)*anchor_width
    h = exp(dh)*anchor_height
        dw and dh -> log-space predictions for width and height (this enables predicting boxes of various sizes.)
    NOTE: Refer original YoloV3 paper: "YoloV3 - an incremental improvement": https://arxiv.org/pdf/1804.02767.pdf for more details on converting offset predictions into original image-scale predictions
    NOTE: These are still grid-scale coordinates (of the particular feature map), so you will require stride (of corresponding spatial scale) to convert them to original image-scale coordinates.
    """
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors    # anchors for the corresponding spatial scale (You can check out the anchor boxes for all 3 spatial scales in yolo_voc_config.py
        self.__nA = len(anchors)    # number of anchors per spatial scale (You can check out the strides for all 3 spatial scales in yolo_voc_config.py)
        self.__nC = nC              # number of classes = 10
        self.__stride = stride      # stride for the corresponding spatial scale


    def forward(self, p):
        """
        Converts FPN predictions p to original image-scale predictions p_d

        Args
        p: Prediction from Feature Pyramid Network (FPN)
            Shape = [batch_size, C', x, x] 
            C'=(Number of anchors per scale)*C  , where C= no. of classes (10) + 4 channels box coordinate offsets (dx,dy,dw,dh) + 1 channel (box confidence)
            x is spatial dimension of the features
            Comprises box coordinate  offsets (dx,dy,dw,dh), box confidence and class probablities (both are logits, and hence in range [-1,1])
            C'[0:2] -> dx,dy
            C'[2:4] -> dw,dh
            C'[4:5] -> box confidence
            C'[5:] -> class probabilities
            NOTE: Box confidence and class probabilities are logits here, and hence they need to be mapped to [0, 1] range.

        Output
        p: Reshaped FPN predictions
            Shape = [batch_size, x, x, num_anchors, C'/num_anchors], where num_anchors=3, which is no. of anchors per scale

        p_d: Original Image-scale box coordinates (x,y,w,h), box confidence and class probabilities (both mapped to range [0,1])        
            Shape = [batch_size, x, x, num_anchors, C'/num_anchors], where num_anchors=3, which is no. of anchors per scale

        Returns
        (p, p_d) as tuple. NOTE: Here, p is reshaped while returning.
        TODO: Complete the below forward pass
        """
        ### START YOUR CODE HERE ###
        raise NotImplementedError
        ### END YOUR CODE HERE ###


def gtid():
    ##############################################################
    # TODO: add correct gtid                                     #
    #                                                            #
    ##############################################################
    # Return gtid
    ##############################################################
    #                      END OF YOUR CODE                      #
    ##############################################################
    return 0