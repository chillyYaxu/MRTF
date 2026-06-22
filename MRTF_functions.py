import rasterio
from rasterio.windows import Window
import numpy as np


def read_raster(raster_path):
    dataset = rasterio.open(raster_path)
    raster_profile = dataset.profile
    raster = dataset.read()
    raster = np.transpose(raster, (1, 2, 0))
    raster = raster.astype(np.dtype(np.float32))
    raster_profile['dtype'] = 'float32'

    return raster, raster_profile

def read_raster_NDVI(raster_path):
    dataset = rasterio.open(raster_path)
    raster_profile = dataset.profile
    raster = dataset.read()
    raster = np.transpose(raster, (1, 2, 0))
    raster = raster.astype(np.dtype(np.float32))
    NDVI = (raster[:,:,-1]-raster[:,:,-2])/(raster[:,:,-1]+raster[:,:,-2]+0.00000001)
    NDVI = np.clip(NDVI, 0, 1)
        # 重塑为 (height, width, 1)
    NDVI = NDVI[:, :, np.newaxis]  # 或 NDVI.reshape(NDVI.shape[0], NDVI.shape[1], 1)
    
    # 更新profile（NDVI是单波段）
    raster_profile['dtype'] = 'float32'
    raster_profile['count'] = 1

    return NDVI, raster_profile

def write_raster(raster, raster_profile, raster_path):
    raster_profile["dtype"] = str(raster.dtype)
    raster_profile["height"] = raster.shape[0]
    raster_profile["width"] = raster.shape[1]
    raster_profile["count"] = raster.shape[2]
    image = np.transpose(raster, (2, 0, 1))
    dataset = rasterio.open(raster_path, mode='w', **raster_profile)
    dataset.write(image)
    dataset.close()


def clip_raster(dataset, row_start, row_stop, col_start, col_stop):
    window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
    transform = dataset.window_transform(window)
    clipped_raster = dataset.read(window=window)
    clipped_raster = np.transpose(clipped_raster, (1, 2, 0))
    clipped_profile = dataset.profile
    clipped_profile.update({'width': col_stop - col_start,
                            'height': row_stop - row_start,
                            'transform': transform})

    return clipped_raster, clipped_profile


def color_composite(image, bands_idx):
    image = np.stack([image[:, :, i] for i in bands_idx], axis=2)
    return image

def color_composite_ma(image, bands_idx):
    image = np.ma.stack([image[:, :, i] for i in bands_idx], axis=2)
    return image


def linear_pct_stretch(img, pct=2, max_out=1, min_out=0):

    def gray_process(gray):
        truncated_down = np.percentile(gray, pct)
        truncated_up = np.percentile(gray, 100 - pct)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        return gray

    bands = []
    for band_idx in range(img.shape[2]):
        band = img[:, :, band_idx]
        band_strch = gray_process(band)
        bands.append(band_strch)
    img_pct_strch = np.stack(bands, axis=2)
    return img_pct_strch

def linear_pct_stretch_ma(img, pct=2, max_out=1, min_out=0):

    def gray_process(gray):
        truncated_down = np.percentile(gray, pct)
        truncated_up = np.percentile(gray, 100 - pct)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        return gray

    out = img.copy()
    for band_idx in range(img.shape[2]):
        band = img.data[:, :, band_idx]
        mask = img.mask[:, :, band_idx]
        band_strch = gray_process(band[~mask])
        out.data[:, :, band_idx][~mask] = band_strch
    return out

if __name__ == "__main__":
    F_tb_path = r"C:\baidunetdiskdownload\journal_1\Tianjin\L_20180109.tif"
    Fit_FC_path = r"C:\baidunetdiskdownload\journal_1\Tianjin\L_20180109_NDVI.tif"
    F_tb, F_tb_profile = read_raster_NDVI(F_tb_path)
    write_raster(F_tb, F_tb_profile, Fit_FC_path)