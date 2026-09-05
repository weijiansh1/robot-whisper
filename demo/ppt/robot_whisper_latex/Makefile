LATEXMK := latexmk
GS := gs
JOB := robot_whisper

.PHONY: all pdf render clean

all: pdf

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -jobname=$(JOB) main.tex

render: pdf
	mkdir -p rendered
	$(GS) -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r144 \
		-sOutputFile=rendered/slide-%02d.png $(JOB).pdf

clean:
	$(LATEXMK) -C -jobname=$(JOB) main.tex

