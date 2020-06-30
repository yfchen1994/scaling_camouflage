## Scaling Attack against Computer Vision Applications
---
### Usage
* python ./attack.py --sourceImg <source image path>
                     --targetImg <target image path>
		     --attackImg <path to save the attack image>
		     --outputImg <output image path>
                     --norm <choose $L_p$ attack norm to use>
		    [--resizeFunc <resizing function>]
		    [--interpolation <interpolation method>]
		    [--penalty <constant $c$ set in the attack>]
		    [--imageFactor <factor used to scale image pixel value to [0,1].]
		    [--help]

### Dependencies installation:
* ./setup.sh
---

### Notes

* The code is running on Python3
* Supported Scaling Functions:
  
      func                  interpolation
    
      cv2.resize         -> cv2.INTER_NEAREST
                            cv2.INTER_LINEAR
     	                    cv2.INTER_CUBIC
	                        cv2.INTER_AREA
	                        cv2.INTER_LANCZOS4
    
      Image.Image.resize -> Image.NEAREST
                            Image.LANCZOS
    	        	        Image.BILINEAR
    	        	        Image.BICUBIC
### References
* [Seeing is Not Believing: Camouflage Attacks on Image Scaling Algorithms (USENIX Security '19)](https://www.usenix.org/conference/usenixsecurity19/presentation/xiao)
* [Scaling Camouflage: Content Disguising Attack Against Computer Vision Applications (IEEE TDSC)](https://ieeexplore.ieee.org/abstract/document/8982037)
