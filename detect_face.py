from main import MTCNNMain
import sys

# if len(sys.argv) < 3:
#     print("python3 detect_face.py <input image path> <output image path>")
#     exit(1)

# img_path = sys.argv[1]
# save_path = sys.argv[2]
# img_path = "/Users/gurushant/Downloads/jeevan.jpg"
img_path = "/Users/gurushant/Desktop/samples/dance2.jpg"
save_path = "/Users/gurushant/Desktop/samples/dance2_out.jpg"
m = MTCNNMain(img_path)
boxes = m.detect_faces()
m.draw_square(boxes,save_path=save_path)