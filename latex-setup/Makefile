PDF=main.pdf
TEX=main.tex

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX)
	latexmk -pdf $(TEX)

clean:
	latexmk -c
