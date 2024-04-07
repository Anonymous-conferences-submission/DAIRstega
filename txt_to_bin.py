def text_to_binary(text):
    byte_array = text.encode()
    binary_array = [format(byte, '08b') for byte in byte_array]
    binary_string = ''.join(binary_array)
    return binary_string

text = "Multimedia"
binary_string = text_to_binary(text)
print(binary_string) 
