import abc
import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class BitLevelCompression:
    """
    Class untuk menyimpan method-method umum pada bit-level compression
    """
    @staticmethod
    def compress_b_bits(numbers, b):
        if b == 0:
            return b""
        packed = bytearray()
        accumulator = 0
        bits_in_acc = 0
        for num in numbers:
            # Mengubah angka menjadi representasi b-bit dan menambahkannya ke accumulator
            val = num & ((1 << b) - 1)
            accumulator = (accumulator << b) | val
            bits_in_acc += b

            # Menunggu sampai accumulator sudah memiliki cukup bit untuk menghasilkan byte (8 bits)
            while bits_in_acc >= 8:
                bits_in_acc -= 8
                byte_val = (accumulator >> bits_in_acc) & 0xFF  # Mengambil 8 bit teratas dari accumulator
                packed.append(byte_val)
            accumulator &= (1 << bits_in_acc) - 1
        # Jika masih ada bit yang tersisa di accumulator setelah memproses semua angka, tambahkan byte terakhir
        if bits_in_acc > 0:
            byte_val = (accumulator << (8 - bits_in_acc)) & 0xFF    # Pad di akhir dengan 0
            packed.append(byte_val)
        return bytes(packed)

    @staticmethod
    def decompress_b_bits(packed_bytes, num_elements, b):
        if b == 0:
            return [0] * num_elements
        numbers = []
        accumulator = 0
        bits_in_acc = 0
        byte_idx = 0
        for _ in range(num_elements):
            # Kalau bits yang ada belum cukup untuk menghasil angka, ambil bits dari bytestream sampai cukup
            while bits_in_acc < b:
                # Kalau masih ada byte yang tersisa di bytestream, tambahkan ke accumulator
                if byte_idx < len(packed_bytes):
                    accumulator = (accumulator << 8) | packed_bytes[byte_idx]
                    byte_idx += 1
                    bits_in_acc += 8
                # Kalau bytestream sudah habis tapi kita masih membutuhkan bit, berarti kita sudah di akhir dan harus pad dengan 0
                else:
                    accumulator = accumulator << (b - bits_in_acc)
                    bits_in_acc = b
            # Ambil b bit teratas dari accumulator sebagai angka berikutnya
            bits_in_acc -= b
            val = (accumulator >> bits_in_acc) & ((1 << b) - 1)
            numbers.append(val)
        return numbers

class OptPForDeltaPostings:
    """
    OptPForDelta (Optimized PForDelta) compression. Melakukan delta encoding
    seperti pada VBEPostings,tetapi kemudian untuk setiap block berukuran 128
    (atau sisa elemen), mencari nilai b optimal untuk block tersebut, dan melakukan
    bit parameter-packing (sebanyak b bits per element) untuk elemen-elemen yang
    tidak termasuk dalam exceptions. Untuk elemen-elemen yang termasuk dalam exceptions,
    simpan indeks dan nilai aslinya secara terpisah. Untuk implementasi ini, exceptions
    disimpan sebagai array 32-bit integer untuk menyederhanakan implementasi
    """
    BLOCK_SIZE = 128

    @staticmethod
    def _encode_sequence(sequence):
        bytestream = bytearray()
        for i in range(0, len(sequence), OptPForDeltaPostings.BLOCK_SIZE):
            block = sequence[i : i + OptPForDeltaPostings.BLOCK_SIZE]
            
            # Cari nilai b optimal untuk block ini
            b = OptPForDeltaPostings._find_optimal_b(block)
            max_val_for_b = (1 << b) - 1
            
            exceptions_indices = []
            exceptions_values = []
            
            # Cari exceptions dan simpan indeks serta nilai aslinya
            for idx, val in enumerate(block):
                if val > max_val_for_b:
                    exceptions_indices.append(idx)
                    exceptions_values.append(val)
            
            num_elements = len(block)
            num_exceptions = len(exceptions_values)
            
            # Block metadata: [num_elements, b, num_exceptions]
            bytestream.append(num_elements)
            bytestream.append(b)
            bytestream.append(num_exceptions)
            
            # Tambahkan data yang sudah dicompress dengan b bits per element
            bytestream.extend(BitLevelCompression.compress_b_bits(block, b))
            
            # Tambahkan informasi exceptions: indeks dan nilai aslinya (jika ada)
            bytestream.extend(array.array('B', exceptions_indices).tobytes())
            if exceptions_values:
                exceptions_arr = array.array('I', exceptions_values)
                bytestream.extend(exceptions_arr.tobytes())
            
        return bytes(bytestream)

    @staticmethod
    def _decode_sequence(encoded_list):
        if not encoded_list:
            return []
            
        decoded = []
        offset = 0
        
        # Ukuran yang digunakan untuk menyimpan exceptions
        itemsize_I = array.array('I').itemsize
        
        while offset < len(encoded_list):
            # Ambil metadata block
            num_elements = encoded_list[offset]
            b = encoded_list[offset + 1]
            num_exceptions = encoded_list[offset + 2]
            offset += 3
            
            # Ambil bytes yang diperlukan untuk block ini dengan pembulatan ke atas
            packed_len = (num_elements * b + 7) // 8
            packed_bytes = encoded_list[offset : offset + packed_len]
            offset += packed_len
            
            # Ambil list index exceptions
            exceptions_indices = list(encoded_list[offset : offset + num_exceptions])
            offset += num_exceptions
            
            # Ambil bytes yang berisi nilai asli exceptions
            exceptions_bytes_len = num_exceptions * itemsize_I
            exceptions_values = []
            if num_exceptions > 0:
                exceptions_values_bytes = encoded_list[offset : offset + exceptions_bytes_len]
                exceptions_arr = array.array('I')
                exceptions_arr.frombytes(exceptions_values_bytes)
                exceptions_values = exceptions_arr.tolist()
            offset += exceptions_bytes_len
            
            # Decompress block ini dengan b bits per element
            numbers = BitLevelCompression.decompress_b_bits(packed_bytes, num_elements, b)
            
            # Pasangkan indeks exceptions dengan nilai asli dan replace nilainya pada list
            for idx, val in zip(exceptions_indices, exceptions_values):
                numbers[idx] = val
                
            decoded.extend(numbers)
            
        return decoded
    
    @staticmethod
    def _find_optimal_b(block, W=32):
        """Mencari nilai b optimal untuk block ini dengan mencoba semua kemungkinan b dari 0 sampai W,
        dan menghitung cost untuk setiap b. Cost dihitung sebagai jumlah bit yang dibutuhkan untuk
        menyimpan semua elemen dalam block dengan b bits per element, ditambah overhead untuk exceptions"""
        best_b, best_cost = 0, float('inf')
        for b in range(W + 1):
            threshold = (1 << b) - 1 if b < W else 0xFFFFFFFF   # nilai maksimum untuk b bits
            e = sum(1 for v in block if v > threshold)  # jumlah exceptions untuk b ini
            cost = len(block) * b + 128 + e * W  # cost = jumlah non-exception * b bit + jumlah exception * 32 bit untuk menyimpannya
            if cost < best_cost:
                best_cost, best_b = cost, b
        return best_b

    @staticmethod
    def encode(postings_list):
        # Gunakan gap encoding seperti pada VBEPostings sebelum melakukan bit packing
        gaps = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i-1])
        return OptPForDeltaPostings._encode_sequence(gaps)

    @staticmethod
    def decode(encoded_postings_list):
        decoded_gaps = OptPForDeltaPostings._decode_sequence(encoded_postings_list)
        if not decoded_gaps:
             return []
        
        # Rekonstruksi postings list asli dari gap-based list
        postings = []
        current = 0
        for gap in decoded_gaps:
            current += gap
            postings.append(current)
            
        return postings

    @staticmethod
    def encode_tf(tf_list):
        return OptPForDeltaPostings._encode_sequence(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        return OptPForDeltaPostings._decode_sequence(encoded_tf_list)

class BP128Postings:
    """
    BP128 (Bit Packing 128) compression.
    Membagi sequence menjadi block berukuran 128 (atau sisa elemen),
    mencari nilai maksimum di block tersebut untuk menentukan bit-width (b),
    dan melakukan bit parameter-packing (sebanyak b bits per elemen).
    """
    BLOCK_SIZE = 128

    @staticmethod
    def _encode_sequence(sequence):
        bytestream = bytearray()
        for i in range(0, len(sequence), BP128Postings.BLOCK_SIZE):
            block = sequence[i : i + BP128Postings.BLOCK_SIZE]
            
            # BP128: Cari maximum bit length dalam block
            b = max(block).bit_length()
            
            num_elements = len(block)
            
            # Tambahkan metadata: [num_elements, b]
            bytestream.append(num_elements)
            bytestream.append(b)
            
            # Lanjut dengan data yang sudah dicompress dengan b bits per element
            bytestream.extend(BitLevelCompression.compress_b_bits(block, b))
            
        return bytes(bytestream)

    @staticmethod
    def _decode_sequence(encoded_list):
        if not encoded_list:
            return []
            
        decoded = []
        offset = 0
        
        while offset < len(encoded_list):
            # Ambil metadata block
            num_elements = encoded_list[offset]
            b = encoded_list[offset + 1]
            offset += 2
            
            # Ambil bytes yang diperlukan untuk block ini dengan pembulatan ke atas
            packed_len = (num_elements * b + 7) // 8
            packed_bytes = encoded_list[offset : offset + packed_len]
            offset += packed_len
            
            # Decompress block ini dengan b bits per element
            numbers = BitLevelCompression.decompress_b_bits(packed_bytes, num_elements, b)
            decoded.extend(numbers)
            
        return decoded

    @staticmethod
    def encode(postings_list):
        if not postings_list:
            return b""
        
        # Gunakan gap encoding seperti pada VBEPostings sebelum melakukan bit packing
        gaps = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i-1])

        return BP128Postings._encode_sequence(gaps)

    @staticmethod
    def decode(encoded_postings_list):
        decoded_gaps = BP128Postings._decode_sequence(encoded_postings_list)
        if not decoded_gaps:
             return []
        
        # Rekonstruksi postings list asli dari gap-based list
        postings = []
        current = 0
        for gap in decoded_gaps:
            current += gap
            postings.append(current)
            
        return postings

    @staticmethod
    def encode_tf(tf_list):
        return BP128Postings._encode_sequence(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        return BP128Postings._decode_sequence(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, OptPForDeltaPostings, BP128Postings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
