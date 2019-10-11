### Usage
* python ./attack.py --sourceImg <source image path>
                     --targetImg <target image path>
                     --attackImg <path to save the attack image>
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
* The version of cvxpy must be 0.4.11
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
