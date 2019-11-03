text = "this is the first line\nthis is the next line\nthis is the last line"
append_text = '\nthis is a appended file'
my_file = open('my file.txt','a')
my_file.write(append_text)
my_file.close()
