import numpy
from PIL import Image, ImageDraw

def nbimage(data):
	from IPython.display import display, Image
	from PIL.Image import fromarray
	from StringIO import StringIO
	s = StringIO()
	data.save(s,'PNG')
	display(Image(s.getvalue()))

WIDTH = 512
HEIGHT = 288

im = Image.new("RGB", (WIDTH, HEIGHT), "black")

draw = ImageDraw.Draw(im)
for i in range(50):
	j, k, l, m = numpy.random.randint(0,WIDTH), numpy.random.randint(0,HEIGHT), numpy.random.randint(0,WIDTH), numpy.random.randint(0,HEIGHT)
	n, o, p = numpy.random.randint(0,256), numpy.random.randint(0,256), numpy.random.randint(0,256)
	switcher = numpy.random.randint(0,4)
	if switcher == 0:
		draw.line((j,k,l,m), (n,o,p))
	elif switcher == 1:
		draw.ellipse((j,k,l,m), (n,o,p))
	elif switcher == 2:
		q,r,s,t,u,v = numpy.random.randint(0,WIDTH), numpy.random.randint(0,HEIGHT), numpy.random.randint(0,WIDTH), numpy.random.randint(0,HEIGHT), numpy.random.randint(0,WIDTH), numpy.random.randint(0,HEIGHT)
		draw.polygon((j,k,l,m,q,r,s,t,u,v), (n,o,p))
	else:
		draw.rectangle((j,k,l,m), (n,o,p))
del draw
nbimage(im)