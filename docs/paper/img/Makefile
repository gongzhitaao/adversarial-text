TEXI2PDF = texi2pdf
FLAGS = --clean --pdf

# find the most recently changed tex file in 1 minute.  If more than
# one files are changed within 1 minute, list only the most recent one.
cur := "$(shell find . -name '*.tex' -mmin -1 -printf '%TT\t%p\n' | sort | tail -1 | awk '{print $$2}')"
cur := "$(shell basename $(cur))"

all : recent

recent :
ifneq ($(cur),"")
	$(TEXI2PDF) $(FLAGS) $(cur)
endif

.PHONY : all recent
