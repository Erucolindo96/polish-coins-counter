import PIL.Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont


class CoinSumDrawer:
    def __init__(self, labels, font, color='red'):
        self.labels = labels
        self.color = color
        self.label_to_value = {
            '1gr': 1,
            '1zl': 100,
            '2gr': 2,
            '2zl': 200,
            '5gr': 5,
            '5zl': 500,
            '10gr': 10,
            '20gr': 20,
            '50gr': 50,
            'non-polish': 0
        }
        self.sum = 0
        self.font = font

    def count(self):
        for label in self.labels:
            self.sum += self.label_to_value[label]

    def draw_sum(self, image: PIL.Image.Image):
        string_to_display = str(float(self.sum) / 100) + ' zł'
        pixel_per_digit = 15
        bbox_width = len(string_to_display) * pixel_per_digit

        sum_bbox = (5, 5, 5 + bbox_width, 60)
        currency_pos = (10, 20)

        draw = PIL.ImageDraw.Draw(image)
        draw.rectangle(sum_bbox, outline=self.color)
        draw.text(currency_pos, string_to_display,
                  font=PIL.ImageFont.truetype(self.font, 20), fill=self.color)
        return image
