from vnstock import Listing

listing = Listing()

df = listing.all_symbols()

# Bảng gồm 2 cột: tên mã cổ phiếu và tên công ty
codes = df['symbol'].tolist()

print(df.columns) 
print(codes)
print("Số lượng:", len(codes))