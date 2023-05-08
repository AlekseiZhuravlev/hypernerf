class Image:
    def __init__(self,

                 ):
        self.img_path = img_path
        self.img = Image.open(img_path)
        self.img_arr = np.array(self.img)
        self.width, self.height = self.img.size