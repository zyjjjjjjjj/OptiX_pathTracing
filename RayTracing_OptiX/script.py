f = open("./RayTracing/x64/Debug/devicePrograms.cu.ptx", 'rb')
w = open("./RayTracing/MyShader.c", 'w', True)
w.write("#ifdef __cplusplus\nextern \"C\" {\n#endif\n")
w.write("const unsigned char embedded_ptx_code[] = {\n")
k = 1
while True:
    ch = f.read(1)
    if not ch:
        break
    w.write('0x{:02X}'.format(int.from_bytes(ch, byteorder='big', signed=False)))
    w.write(",")
    if k % 16 == 0:
        w.write("\n")
    k += 1
f.close()
w.write("};\n#ifdef __cplusplus\n}\n#endif")
