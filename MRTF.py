from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.transform import resize
from MRTF_functions import *
from skimage.transform import downscale_local_mean
from scipy.spatial import cKDTree
from scipy.ndimage import sobel
from datetime import datetime
from scipy.ndimage import uniform_filter
from sklearn.linear_model import HuberRegressor
from numba import jit, prange

from scipy.spatial.distance import cdist




class proposed:
    def __init__(self, F_t1, C_t1, C_t2, RM_win_size=20, scale_factor=16, similar_win_size=20, similar_num=20):
        self.F_t1 = F_t1.astype(np.float32)
        self.C_t1 = C_t1.astype(np.float32)
        self.C_t2 = C_t2.astype(np.float32)
        self.RM_win_size = RM_win_size
        self.scale_factor = scale_factor
        self.similar_win_size = similar_win_size
        self.similar_num = similar_num

    
    
    def classify_reference_images_fast_v2(self, ref_image, class_num):
        """
        快速图像分类
        """
        
        h, w, c = ref_image.shape
        
        print(f"  Classifying image with {h}x{w} pixels, {c} bands...")
        
        # 重塑数据
        X = np.reshape(ref_image, (h * w, c))
        
        # 简单的标准化（避免使用StandardScaler）
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True) + 1e-8
        X_normalized = (X - X_mean) / X_std
        
        # 使用MiniBatchKMeans，效率高且适合大图像
        batch_size = min(1000, h * w // 10)
        
        kmeans = MiniBatchKMeans(
            n_clusters=class_num,
            batch_size=batch_size,
            random_state=42,
            n_init=3,
            max_iter=100,
            verbose=0
        )
        
        # 如果图像太大，使用部分数据进行训练
        if h * w > 1000000:
            # 随机采样10%的数据进行训练
            sample_size = min(5000000, h * w // 10)
            indices = np.random.choice(h * w, sample_size, replace=False)
            X_sample = X_normalized[indices]
            
            print(f"    Training on {sample_size} samples...")
            kmeans.fit(X_sample)
            
            # 预测所有像素
            print(f"    Predicting for all {h*w} pixels...")
            class_map = kmeans.predict(X_normalized)
        else:
            # 直接训练和预测
            print(f"    Training on all {h*w} pixels...")
            class_map = kmeans.fit_predict(X_normalized)
        
        class_map = class_map.reshape((h, w))
        
        # 统计每个类别的像素数
        unique, counts = np.unique(class_map, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    Class {u}: {c} pixels ({c/(h*w)*100:.1f}%)")
        
        return class_map
    


    def regression_model_fitting_combined(self, band_idx, scales=[1, 2, 4], 
                                        non_linear_ratio=0.3, sigma=0.12,
                                        min_val=None, max_val=None):
        
        C_t1_band = self.C_t1[:, :, band_idx]
        C_t2_band = self.C_t2[:, :, band_idx]
        
        # 自动检测数据范围
        if min_val is None:
            min_val = min(np.nanmin(C_t1_band), np.nanmin(C_t2_band))
        if max_val is None:
            max_val = max(np.nanmax(C_t1_band), np.nanmax(C_t2_band))
        
        data_range = max_val - min_val
        
        # 归一化到 [-1, 1] 范围
        if data_range > 1e-8:
            C_t1_norm = 2.0 * (C_t1_band - min_val) / data_range - 1.0
            C_t2_norm = 2.0 * (C_t2_band - min_val) / data_range - 1.0
        else:
            C_t1_norm = np.zeros_like(C_t1_band)
            C_t2_norm = np.zeros_like(C_t2_band)
        
        h, w = C_t1_norm.shape
        win_size = self.RM_win_size
        
        # 自适应sigma（基于数据范围）
        sigma_adaptive = sigma * data_range if data_range > 1e-8 else sigma
        
        print(f"\n{'='*60}")
        print(f"Combined Regression for Band {band_idx}")
        print(f"{'='*60}")
        print(f"  Input range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  Data range: {data_range:.4f}")
        print(f"  Normalized range: [-1.0, 1.0]")
        print(f"  Multi-Scale Scales: {scales}")
        print(f"  Non-linear ratio: {non_linear_ratio}")
        
        # ============ 1. 局部纹理复杂度（用于多尺度权重） ============
        gy, gx = np.gradient(C_t1_norm)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        local_texture = uniform_filter(gradient_mag, size=5, mode='reflect')
        
        t_max = np.percentile(local_texture, 95)
        texture_norm = np.clip(local_texture / max(t_max, 1e-6), 0, 1)
        
        print(f"  Texture range: [{np.min(texture_norm):.3f}, {np.max(texture_norm):.3f}]")
        
        # ============ 预计算多尺度图像（包含纹理） ============
        scaled_images = {}
        for scale in scales:
            if scale > 1:
                h_scale, w_scale = h // scale, w // scale
                C_t1_scale = resize(C_t1_norm, (h_scale, w_scale), order=1, preserve_range=True)
                C_t2_scale = resize(C_t2_norm, (h_scale, w_scale), order=1, preserve_range=True)
                texture_scale = resize(texture_norm, (h_scale, w_scale), order=1, preserve_range=True)
                scaled_images[scale] = (C_t1_scale, C_t2_scale, texture_scale, h_scale, w_scale)
            else:
                scaled_images[scale] = (C_t1_norm, C_t2_norm, texture_norm, h, w)
        
        # ============ 2. 多尺度非线性回归 ============
        print("\n  [Part A] Computing Multi-Scale Nonlinear Regression...")
        
        @jit(nopython=True, parallel=True)
        def fast_quadratic_loop(C_t1_scale, C_t2_scale, h_scale, w_scale,
                                scale_win, pad_size, global_t2_mean, bandwidth):
            """局部加权二次多项式回归"""
            pred = np.zeros((h_scale, w_scale), dtype=np.float32)
            win_area = scale_win * scale_win
            
            for i in prange(h_scale):
                x_win = np.zeros(win_area, dtype=np.float32)
                y_win = np.zeros(win_area, dtype=np.float32)
                
                for j in range(w_scale):
                    n_valid = 0
                    for ii in range(scale_win):
                        row = i + ii
                        for jj in range(scale_win):
                            col = j + jj
                            x_val = C_t1_scale[row, col]
                            y_val = C_t2_scale[row, col]
                            
                            if not np.isnan(x_val) and not np.isnan(y_val):
                                x_win[n_valid] = x_val
                                y_win[n_valid] = y_val
                                n_valid += 1
                    
                    center_x = C_t1_scale[i + pad_size, j + pad_size]
                    
                    if n_valid >= 6:
                        # 计算自适应带宽
                        x_range = np.max(x_win[:n_valid]) - np.min(x_win[:n_valid])
                        h = max(bandwidth, x_range * 0.1)
                        
                        # 构建加权最小二乘
                        X = np.zeros((n_valid, 3), dtype=np.float32)
                        W = np.zeros((n_valid, n_valid), dtype=np.float32)
                        y = y_win[:n_valid]
                        
                        for k in range(n_valid):
                            xk = x_win[k]
                            X[k, 0] = 1.0
                            X[k, 1] = xk
                            X[k, 2] = xk * xk
                            
                            dist = abs(xk - center_x)
                            if dist <= h:
                                weight = (1.0 - (dist / h)) ** 3
                            else:
                                weight = 1e-6
                            W[k, k] = weight
                        
                        XtW = np.dot(X.T, W)
                        XtWX = np.dot(XtW, X)
                        XtWy = np.dot(XtW, y)
                        
                        for k in range(3):
                            XtWX[k, k] += 1e-6
                        
                        try:
                            coeffs = np.linalg.solve(XtWX, XtWy)
                            pred_val = coeffs[0] + coeffs[1] * center_x + coeffs[2] * center_x * center_x
                        except:
                            pred_val = np.mean(y)
                    elif n_valid >= 3:
                        xv = x_win[:n_valid]
                        yv = y_win[:n_valid]
                        x_mean = np.mean(xv)
                        y_mean = np.mean(yv)
                        numerator = np.sum((xv - x_mean) * (yv - y_mean))
                        denominator = np.sum((xv - x_mean)**2)
                        
                        if abs(denominator) > 1e-8:
                            slope = numerator / denominator
                            intercept = y_mean - slope * x_mean
                            pred_val = intercept + slope * center_x
                        else:
                            pred_val = y_mean
                    else:
                        pred_val = global_t2_mean if n_valid == 0 else np.mean(y_win[:n_valid])
                    
                    pred[i, j] = max(-1.0, min(1.0, pred_val))
            
            return pred
        
        # 计算各尺度非线性预测
        nonlinear_preds_list = []
        nonlinear_texture_list = []
        
        for scale in scales:
            C_t1_scale, C_t2_scale, texture_scale, h_scale, w_scale = scaled_images[scale]
            
            scale_win = max(win_size // scale, 5)
            pad_size = scale_win // 2
            
            C_t1_pad_scale = np.pad(C_t1_scale, pad_size, mode='reflect')
            C_t2_pad_scale = np.pad(C_t2_scale, pad_size, mode='reflect')
            
            mask_scale = ~np.isnan(C_t2_scale)
            global_t2_mean_scale = np.mean(C_t2_scale[mask_scale]) if mask_scale.any() else 0.0
            
            bandwidth_scale = sigma_adaptive * np.sqrt(scale)
            
            scale_pred = fast_quadratic_loop(
                C_t1_pad_scale, C_t2_pad_scale, h_scale, w_scale,
                scale_win, pad_size, global_t2_mean_scale, bandwidth_scale
            )
            
            if scale > 1:
                pred_up = resize(scale_pred, (h, w), order=1, preserve_range=True)
                texture_up = resize(texture_scale, (h, w), order=1, preserve_range=True)
                nonlinear_preds_list.append(pred_up)
                nonlinear_texture_list.append(texture_up)
            else:
                nonlinear_preds_list.append(scale_pred)
                nonlinear_texture_list.append(texture_scale)
        
        # 多尺度非线性融合：使用纹理权重
        n_scales = len(scales)
        if n_scales == 1:
            nonlinear_fused_norm = nonlinear_preds_list[0]
        elif n_scales == 2:
            # 小尺度权重 = 纹理强度，大尺度权重 = 1-纹理强度
            weights = [texture_norm, 1.0 - texture_norm]
            nonlinear_fused_norm = weights[0] * nonlinear_preds_list[0] + weights[1] * nonlinear_preds_list[1]
        else:
            # 三尺度融合：小尺度捕捉细节（纹理区域权重高）
            #           大尺度捕捉结构（平滑区域权重高）
            #           中尺度过渡
            w_small = texture_norm
            w_medium = 1.0 - 2.0 * np.abs(texture_norm - 0.5)
            w_medium = np.maximum(w_medium, 0.0)
            w_large = 1.0 - texture_norm
            total_w = w_small + w_medium + w_large + 1e-8
            nonlinear_fused_norm = (w_small * nonlinear_preds_list[0] + 
                                    w_medium * nonlinear_preds_list[1] + 
                                    w_large * nonlinear_preds_list[2]) / total_w
        
        nonlinear_fused_norm = np.clip(nonlinear_fused_norm, -1.0, 1.0)
        
        # ============ 3. 多尺度线性Huber回归 ============
        print("\n  [Part B] Computing Multi-Scale Linear Regression (Huber)...")
        
        linear_preds_list = []
        
        @jit(nopython=True, parallel=True)
        def fast_linear_huber_loop(C_t1_scale, C_t2_scale, h_scale, w_scale,
                                    scale_win, pad_size, global_t2_mean, delta):
            """多尺度线性Huber回归"""
            scale_pred = np.zeros((h_scale, w_scale), dtype=np.float32)
            win_area = scale_win * scale_win
            
            for i in prange(h_scale):
                x_win = np.zeros(win_area, dtype=np.float32)
                y_win = np.zeros(win_area, dtype=np.float32)
                
                for j in range(w_scale):
                    n_valid = 0
                    for ii in range(scale_win):
                        row = i + ii
                        for jj in range(scale_win):
                            col = j + jj
                            x_val = C_t1_scale[row, col]
                            y_val = C_t2_scale[row, col]
                            
                            if not np.isnan(x_val) and not np.isnan(y_val):
                                x_win[n_valid] = x_val
                                y_win[n_valid] = y_val
                                n_valid += 1
                    
                    center_x = C_t1_scale[i + pad_size, j + pad_size]
                    
                    if n_valid >= 5:
                        # OLS初始估计
                        x_sum = 0.0
                        y_sum = 0.0
                        xy_sum = 0.0
                        x2_sum = 0.0
                        
                        for k in range(n_valid):
                            xk = x_win[k]
                            yk = y_win[k]
                            x_sum += xk
                            y_sum += yk
                            xy_sum += xk * yk
                            x2_sum += xk * xk
                        
                        x_mean = x_sum / n_valid
                        y_mean = y_sum / n_valid
                        denominator = x2_sum - x_sum * x_sum / n_valid
                        
                        if abs(denominator) > 1e-10:
                            slope = (xy_sum - x_sum * y_sum / n_valid) / denominator
                            intercept = y_mean - slope * x_mean
                        else:
                            slope = 0.0
                            intercept = y_mean
                        
                        # Huber加权优化
                        w_sum = 0.0
                        wx_sum = 0.0
                        wy_sum = 0.0
                        wxy_sum = 0.0
                        wx2_sum = 0.0
                        
                        for k in range(n_valid):
                            residual = y_win[k] - (intercept + slope * x_win[k])
                            abs_res = abs(residual)
                            if abs_res > delta:
                                w = delta / abs_res
                            else:
                                w = 1.0
                            
                            xk = x_win[k]
                            yk = y_win[k]
                            w_sum += w
                            wx_sum += w * xk
                            wy_sum += w * yk
                            wxy_sum += w * xk * yk
                            wx2_sum += w * xk * xk
                        
                        if w_sum > 1e-10:
                            w_denom = wx2_sum - wx_sum * wx_sum / w_sum
                            if abs(w_denom) > 1e-10:
                                slope = (wxy_sum - wx_sum * wy_sum / w_sum) / w_denom
                                intercept = (wy_sum - slope * wx_sum) / w_sum
                        
                        pred_val = intercept + slope * center_x
                    else:
                        pred_val = global_t2_mean
                    
                    scale_pred[i, j] = max(-1.0, min(1.0, pred_val))
            
            return scale_pred
        
        # 计算各尺度线性预测
        for scale in scales:
            C_t1_scale, C_t2_scale, texture_scale, h_scale, w_scale = scaled_images[scale]
            
            scale_win = max(win_size // scale, 5)
            pad_size = scale_win // 2
            
            C_t1_pad_scale = np.pad(C_t1_scale, pad_size, mode='reflect')
            C_t2_pad_scale = np.pad(C_t2_scale, pad_size, mode='reflect')
            
            mask_scale = ~np.isnan(C_t2_scale)
            global_t2_mean_scale = np.mean(C_t2_scale[mask_scale]) if mask_scale.any() else 0.0
            
            delta = 0.1
            
            scale_pred = fast_linear_huber_loop(
                C_t1_pad_scale, C_t2_pad_scale, h_scale, w_scale,
                scale_win, pad_size, global_t2_mean_scale, delta
            )
            
            if scale > 1:
                pred_up = resize(scale_pred, (h, w), order=1, preserve_range=True)
                linear_preds_list.append(pred_up)
            else:
                linear_preds_list.append(scale_pred)
        
        # 多尺度线性融合：同样使用纹理权重
        if n_scales == 1:
            linear_fused_norm = linear_preds_list[0]
        elif n_scales == 2:
            weights = [texture_norm, 1.0 - texture_norm]
            linear_fused_norm = weights[0] * linear_preds_list[0] + weights[1] * linear_preds_list[1]
        else:
            w_small = texture_norm
            w_medium = 1.0 - 2.0 * np.abs(texture_norm - 0.5)
            w_medium = np.maximum(w_medium, 0.0)
            w_large = 1.0 - texture_norm
            total_w = w_small + w_medium + w_large + 1e-8
            linear_fused_norm = (w_small * linear_preds_list[0] + 
                                w_medium * linear_preds_list[1] + 
                                w_large * linear_preds_list[2]) / total_w
        
        linear_fused_norm = np.clip(linear_fused_norm, -1.0, 1.0)
        
        # ============ 4. 自适应融合（最终融合） ============
        print("\n  [Part C] Adaptive Fusion...")
        
        # 最终融合：根据纹理强度决定线性和非线性的比例
        # 纹理强的区域非线性权重高，纹理弱的区域线性权重高
        final_adaptive_weight = non_linear_ratio * (0.5 + 0.5 * texture_norm)
        
        combined_pred_norm = (1 - final_adaptive_weight) * linear_fused_norm + final_adaptive_weight * nonlinear_fused_norm
        combined_pred_norm = np.clip(combined_pred_norm, -1.0, 1.0)
        
        print(f"  Final weight range: [{np.min(final_adaptive_weight):.3f}, {np.max(final_adaptive_weight):.3f}]")
        
        # ============ 5. 反归一化 ============
        if data_range > 1e-8:
            combined_pred = (combined_pred_norm + 1.0) / 2.0 * data_range + min_val
            linear_fused = (linear_fused_norm + 1.0) / 2.0 * data_range + min_val
            nonlinear_fused = (nonlinear_fused_norm + 1.0) / 2.0 * data_range + min_val
        else:
            combined_pred = combined_pred_norm
            linear_fused = linear_fused_norm
            nonlinear_fused = nonlinear_fused_norm
        
        combined_pred = np.clip(combined_pred, min_val, max_val)
        
        # 输出统计
        linear_mse = np.mean((C_t2_band - linear_fused)**2)
        nonlinear_mse = np.mean((C_t2_band - nonlinear_fused)**2)
        combined_mse = np.mean((C_t2_band - combined_pred)**2)
        
        print(f"\n  Performance Comparison:")
        print(f"    Linear MSE: {linear_mse:.6f}")
        print(f"    Nonlinear MSE: {nonlinear_mse:.6f}")
        print(f"    Combined MSE: {combined_mse:.6f}")
        print(f"    Improvement: {(linear_mse - combined_mse) / linear_mse * 100:.2f}%")
        
        # ============ 6. 反推线性系数 ============
        coeffs_fused = np.zeros((h, w, 2), dtype=np.float32)
        half_win = win_size // 2
        
        for i in range(h):
            i_min = max(0, i - half_win)
            i_max = min(h, i + half_win + 1)
            for j in range(w):
                j_min = max(0, j - half_win)
                j_max = min(w, j + half_win + 1)
                
                x_local = C_t1_band[i_min:i_max, j_min:j_max].flatten()
                y_local = combined_pred[i_min:i_max, j_min:j_max].flatten()
                
                valid = ~np.isnan(x_local) & ~np.isnan(y_local)
                x_valid = x_local[valid]
                y_valid = y_local[valid]
                
                if len(x_valid) >= 5:
                    x_mean = np.mean(x_valid)
                    y_mean = np.mean(y_valid)
                    
                    numerator = np.sum((x_valid - x_mean) * (y_valid - y_mean))
                    denominator = np.sum((x_valid - x_mean)**2)
                    
                    if abs(denominator) > 1e-10:
                        slope = numerator / denominator
                        intercept = y_mean - slope * x_mean
                    else:
                        slope = 0.0
                        intercept = y_mean
                    
                    slope = np.clip(slope, -2.0, 2.0)
                    intercept = np.clip(intercept, min_val, max_val)
                    
                    coeffs_fused[i, j] = [intercept, slope]
                else:
                    coeffs_fused[i, j, 0] = np.clip(combined_pred[i, j], min_val, max_val)
                    coeffs_fused[i, j, 1] = 0.0
        
        r = C_t2_band - combined_pred
        
        return coeffs_fused, r

    def regression_model_fitting_combined_v2(self, band_idx, scales=[1, 2, 4], 
                                       non_linear_ratio=0.3, sigma=0.12,
                                       min_val=None, max_val=None):
        """
        组合回归：多尺度线性(Huber) + 多尺度非线性回归（局部加权多项式）
        
        核心思想：
        1. 多尺度线性Huber回归：捕捉全局线性趋势，对异常值鲁棒
        2. 多尺度非线性回归：使用局部加权二次多项式，捕捉非线性细节
        3. 自适应融合：根据纹理复杂度动态调整两者权重
        4. 严格的范围限制：确保预测值在有效范围内
        
        Parameters:
        -----------
        band_idx : int
            波段索引
        scales : list
            多尺度回归的尺度因子列表
        non_linear_ratio : float
            非线性部分的基础权重 (0-1)
        sigma : float
            局部回归的带宽参数
        min_val : float or None
            数据最小值，如果为None则自动从数据中检测
        max_val : float or None
            数据最大值，如果为None则自动从数据中检测
        
        Returns:
        --------
        coeffs_fused : array (h, w, 2)
            拟合系数 [intercept, slope]
        r : array (h, w)
            残差
        combined_pred : array (h, w)
            组合预测值（已限制范围）
        """
        
        C_t1_band = self.C_t1[:, :, band_idx]
        C_t2_band = self.C_t2[:, :, band_idx]
        
        # 自动检测数据范围
        if min_val is None:
            min_val = min(np.nanmin(C_t1_band), np.nanmin(C_t2_band))
        if max_val is None:
            max_val = max(np.nanmax(C_t1_band), np.nanmax(C_t2_band))
        
        data_range = max_val - min_val
        
        # 归一化到 [-1, 1] 范围
        if data_range > 1e-8:
            C_t1_norm = 2.0 * (C_t1_band - min_val) / data_range - 1.0
            C_t2_norm = 2.0 * (C_t2_band - min_val) / data_range - 1.0
        else:
            C_t1_norm = np.zeros_like(C_t1_band)
            C_t2_norm = np.zeros_like(C_t2_band)
        
        h, w = C_t1_norm.shape
        win_size = self.RM_win_size
        
        # 自适应sigma（基于数据范围）
        sigma_adaptive = sigma * data_range if data_range > 1e-8 else sigma
        
        print(f"\n{'='*60}")
        print(f"Combined Regression for Band {band_idx}")
        print(f"{'='*60}")
        print(f"  Input range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  Data range: {data_range:.4f}")
        print(f"  Normalized range: [-1.0, 1.0]")
        print(f"  Multi-Scale Scales: {scales}")
        print(f"  Non-linear ratio: {non_linear_ratio}")
        
        # ============ 1. 局部纹理复杂度（仅用于adaptive_weight） ============
        gy, gx = np.gradient(C_t1_norm)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        local_texture = uniform_filter(gradient_mag, size=5, mode='reflect')
        
        t_max = np.percentile(local_texture, 95)
        texture_norm = np.clip(local_texture / max(t_max, 1e-6), 0, 1)
        
        # 自适应融合权重
        adaptive_weight = non_linear_ratio * (0.5 + 0.5 * texture_norm)
        
        print(f"  Texture range: [{np.min(texture_norm):.3f}, {np.max(texture_norm):.3f}]")
        print(f"  Weight range: [{np.min(adaptive_weight):.3f}, {np.max(adaptive_weight):.3f}]")
        
        # ============ 预计算多尺度图像 ============
        scaled_images = {}
        for scale in scales:
            if scale > 1:
                h_scale, w_scale = h // scale, w // scale
                C_t1_scale = resize(C_t1_norm, (h_scale, w_scale), order=1, preserve_range=True)
                C_t2_scale = resize(C_t2_norm, (h_scale, w_scale), order=1, preserve_range=True)
                scaled_images[scale] = (C_t1_scale, C_t2_scale, h_scale, w_scale)
            else:
                scaled_images[scale] = (C_t1_norm, C_t2_norm, h, w)
        
        # ============ 2. 多尺度非线性回归（局部加权二次多项式） ============
        print("\n  [Part A] Computing Multi-Scale Nonlinear Regression (Local Quadratic)...")
        
        @jit(nopython=True, parallel=True)
        def fast_quadratic_loop(C_t1_scale, C_t2_scale, h_scale, w_scale,
                            scale_win, pad_size, global_t2_mean, bandwidth):
            """
            局部加权二次多项式回归
            f(x) = a + b*x + c*x^2
            使用三线性核权重
            """
            pred = np.zeros((h_scale, w_scale), dtype=np.float32)
            win_area = scale_win * scale_win
            
            for i in prange(h_scale):
                x_win = np.zeros(win_area, dtype=np.float32)
                y_win = np.zeros(win_area, dtype=np.float32)
                
                for j in range(w_scale):
                    n_valid = 0
                    for ii in range(scale_win):
                        row = i + ii
                        for jj in range(scale_win):
                            col = j + jj
                            x_val = C_t1_scale[row, col]
                            y_val = C_t2_scale[row, col]
                            
                            if not np.isnan(x_val) and not np.isnan(y_val):
                                x_win[n_valid] = x_val
                                y_win[n_valid] = y_val
                                n_valid += 1
                    
                    center_x = C_t1_scale[i + pad_size, j + pad_size]
                    
                    if n_valid >= 6:  # 二次多项式需要至少6个点
                        # 构建加权最小二乘
                        # 设计矩阵: [1, x, x^2]
                        X = np.zeros((n_valid, 3), dtype=np.float32)
                        W = np.zeros((n_valid, n_valid), dtype=np.float32)
                        y = y_win[:n_valid]
                        
                        # 计算三线性核权重
                        x_range = np.max(x_win[:n_valid]) - np.min(x_win[:n_valid])
                        h = max(bandwidth, x_range * 0.1)  # 自适应带宽
                        
                        for k in range(n_valid):
                            xk = x_win[k]
                            X[k, 0] = 1.0
                            X[k, 1] = xk
                            X[k, 2] = xk * xk
                            
                            # 三线性核权重
                            dist = abs(xk - center_x)
                            if dist <= h:
                                weight = (1.0 - (dist / h)) ** 3
                            else:
                                weight = 1e-6
                            W[k, k] = weight
                        
                        # 求解加权最小二乘 (X^T W X) beta = X^T W y
                        XtW = np.dot(X.T, W)
                        XtWX = np.dot(XtW, X)
                        XtWy = np.dot(XtW, y)
                        
                        # 添加小的正则化项
                        for k in range(3):
                            XtWX[k, k] += 1e-6
                        
                        try:
                            coeffs = np.linalg.solve(XtWX, XtWy)
                            # 预测: a + b*x + c*x^2
                            pred_val = coeffs[0] + coeffs[1] * center_x + coeffs[2] * center_x * center_x
                        except:
                            # 失败时使用局部均值
                            pred_val = np.mean(y)
                    elif n_valid >= 3:
                        # 点数不足，使用线性拟合
                        xv = x_win[:n_valid]
                        yv = y_win[:n_valid]
                        x_mean = np.mean(xv)
                        y_mean = np.mean(yv)
                        numerator = np.sum((xv - x_mean) * (yv - y_mean))
                        denominator = np.sum((xv - x_mean)**2)
                        
                        if abs(denominator) > 1e-8:
                            slope = numerator / denominator
                            intercept = y_mean - slope * x_mean
                            pred_val = intercept + slope * center_x
                        else:
                            pred_val = y_mean
                    else:
                        pred_val = global_t2_mean if n_valid == 0 else np.mean(y_win[:n_valid])
                    
                    # 严格限制在 [-1, 1] 范围内
                    pred[i, j] = max(-1.0, min(1.0, pred_val))
            
            return pred
        
        # 计算各尺度非线性预测
        nonlinear_preds_list = []
        
        for scale in scales:
            C_t1_scale, C_t2_scale, h_scale, w_scale = scaled_images[scale]
            
            scale_win = max(win_size // scale, 5)
            pad_size = scale_win // 2
            
            C_t1_pad_scale = np.pad(C_t1_scale, pad_size, mode='reflect')
            C_t2_pad_scale = np.pad(C_t2_scale, pad_size, mode='reflect')
            
            mask_scale = ~np.isnan(C_t2_scale)
            global_t2_mean_scale = np.mean(C_t2_scale[mask_scale]) if mask_scale.any() else 0.0
            
            # 自适应带宽
            bandwidth_scale = sigma_adaptive * np.sqrt(scale)
            
            scale_pred = fast_quadratic_loop(
                C_t1_pad_scale, C_t2_pad_scale, h_scale, w_scale,
                scale_win, pad_size, global_t2_mean_scale, bandwidth_scale
            )
            
            if scale > 1:
                pred_up = resize(scale_pred, (h, w), order=1, preserve_range=True)
                nonlinear_preds_list.append(pred_up)
            else:
                nonlinear_preds_list.append(scale_pred)
        
        # 多尺度非线性融合：简单平均
        nonlinear_fused_norm = np.mean(nonlinear_preds_list, axis=0)
        nonlinear_fused_norm = np.clip(nonlinear_fused_norm, -1.0, 1.0)
        
        # ============ 3. 多尺度线性Huber回归 ============
        print("\n  [Part B] Computing Multi-Scale Linear Regression (Huber)...")
        
        linear_preds_list = []
        
        @jit(nopython=True, parallel=True)
        def fast_linear_huber_loop(C_t1_scale, C_t2_scale, h_scale, w_scale,
                                scale_win, pad_size, global_t2_mean, delta):
            """多尺度线性Huber回归循环，带范围限制"""
            scale_pred = np.zeros((h_scale, w_scale), dtype=np.float32)
            win_area = scale_win * scale_win
            
            for i in prange(h_scale):
                x_win = np.zeros(win_area, dtype=np.float32)
                y_win = np.zeros(win_area, dtype=np.float32)
                
                for j in range(w_scale):
                    n_valid = 0
                    for ii in range(scale_win):
                        row = i + ii
                        for jj in range(scale_win):
                            col = j + jj
                            x_val = C_t1_scale[row, col]
                            y_val = C_t2_scale[row, col]
                            
                            if not np.isnan(x_val) and not np.isnan(y_val):
                                x_win[n_valid] = x_val
                                y_win[n_valid] = y_val
                                n_valid += 1
                    
                    center_x = C_t1_scale[i + pad_size, j + pad_size]
                    
                    if n_valid >= 5:
                        # OLS初始估计
                        x_sum = 0.0
                        y_sum = 0.0
                        xy_sum = 0.0
                        x2_sum = 0.0
                        
                        for k in range(n_valid):
                            xk = x_win[k]
                            yk = y_win[k]
                            x_sum += xk
                            y_sum += yk
                            xy_sum += xk * yk
                            x2_sum += xk * xk
                        
                        x_mean = x_sum / n_valid
                        y_mean = y_sum / n_valid
                        denominator = x2_sum - x_sum * x_sum / n_valid
                        
                        if abs(denominator) > 1e-10:
                            slope = (xy_sum - x_sum * y_sum / n_valid) / denominator
                            intercept = y_mean - slope * x_mean
                        else:
                            slope = 0.0
                            intercept = y_mean
                        
                        # Huber加权优化
                        w_sum = 0.0
                        wx_sum = 0.0
                        wy_sum = 0.0
                        wxy_sum = 0.0
                        wx2_sum = 0.0
                        
                        for k in range(n_valid):
                            residual = y_win[k] - (intercept + slope * x_win[k])
                            abs_res = abs(residual)
                            if abs_res > delta:
                                w = delta / abs_res
                            else:
                                w = 1.0
                            
                            xk = x_win[k]
                            yk = y_win[k]
                            w_sum += w
                            wx_sum += w * xk
                            wy_sum += w * yk
                            wxy_sum += w * xk * yk
                            wx2_sum += w * xk * xk
                        
                        if w_sum > 1e-10:
                            w_denom = wx2_sum - wx_sum * wx_sum / w_sum
                            if abs(w_denom) > 1e-10:
                                slope = (wxy_sum - wx_sum * wy_sum / w_sum) / w_denom
                                intercept = (wy_sum - slope * wx_sum) / w_sum
                        
                        pred_val = intercept + slope * center_x
                    else:
                        pred_val = global_t2_mean
                    
                    # 严格限制在 [-1, 1] 范围内
                    scale_pred[i, j] = max(-1.0, min(1.0, pred_val))
            
            return scale_pred
        
        # 计算各尺度线性预测
        for scale in scales:
            C_t1_scale, C_t2_scale, h_scale, w_scale = scaled_images[scale]
            
            scale_win = max(win_size // scale, 5)
            pad_size = scale_win // 2
            
            C_t1_pad_scale = np.pad(C_t1_scale, pad_size, mode='reflect')
            C_t2_pad_scale = np.pad(C_t2_scale, pad_size, mode='reflect')
            
            mask_scale = ~np.isnan(C_t2_scale)
            global_t2_mean_scale = np.mean(C_t2_scale[mask_scale]) if mask_scale.any() else 0.0
            
            delta = 0.1  # 归一化空间的10%
            
            scale_pred = fast_linear_huber_loop(
                C_t1_pad_scale, C_t2_pad_scale, h_scale, w_scale,
                scale_win, pad_size, global_t2_mean_scale, delta
            )
            
            if scale > 1:
                pred_up = resize(scale_pred, (h, w), order=1, preserve_range=True)
                linear_preds_list.append(pred_up)
            else:
                linear_preds_list.append(scale_pred)
        
        # 多尺度线性融合：简单平均
        linear_fused_norm = np.mean(linear_preds_list, axis=0)
        linear_fused_norm = np.clip(linear_fused_norm, -1.0, 1.0)
        
        # ============ 4. 自适应融合 ============
        print("\n  [Part C] Adaptive Fusion...")
        
        combined_pred_norm = (1 - adaptive_weight) * linear_fused_norm + adaptive_weight * nonlinear_fused_norm
        combined_pred_norm = np.clip(combined_pred_norm, -1.0, 1.0)
        
        # ============ 5. 反归一化回原始范围 ============
        if data_range > 1e-8:
            combined_pred = (combined_pred_norm + 1.0) / 2.0 * data_range + min_val
            linear_fused = (linear_fused_norm + 1.0) / 2.0 * data_range + min_val
            nonlinear_fused = (nonlinear_fused_norm + 1.0) / 2.0 * data_range + min_val
        else:
            combined_pred = combined_pred_norm
            linear_fused = linear_fused_norm
            nonlinear_fused = nonlinear_fused_norm
        
        # 再次限制到原始范围
        combined_pred = np.clip(combined_pred, min_val, max_val)
        
        # 输出统计信息
        linear_mse = np.mean((C_t2_band - linear_fused)**2)
        nonlinear_mse = np.mean((C_t2_band - nonlinear_fused)**2)
        combined_mse = np.mean((C_t2_band - combined_pred)**2)
        
        print(f"\n  Performance Comparison:")
        print(f"    Linear MSE: {linear_mse:.6f}")
        print(f"    Nonlinear MSE: {nonlinear_mse:.6f}")
        print(f"    Combined MSE: {combined_mse:.6f}")
        print(f"    Improvement: {(linear_mse - combined_mse) / linear_mse * 100:.2f}%")
        
        # ============ 6. 反推线性系数 ============
        coeffs_fused = np.zeros((h, w, 2), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                i_min = max(0, i - win_size//2)
                i_max = min(h, i + win_size//2 + 1)
                j_min = max(0, j - win_size//2)
                j_max = min(w, j + win_size//2 + 1)
                
                x_local = C_t1_band[i_min:i_max, j_min:j_max].flatten()
                y_local = combined_pred[i_min:i_max, j_min:j_max].flatten()
                
                valid = ~np.isnan(x_local) & ~np.isnan(y_local)
                x_valid = x_local[valid]
                y_valid = y_local[valid]
                
                if len(x_valid) >= 5:
                    x_mean = np.mean(x_valid)
                    y_mean = np.mean(y_valid)
                    
                    numerator = np.sum((x_valid - x_mean) * (y_valid - y_mean))
                    denominator = np.sum((x_valid - x_mean)**2)
                    
                    if abs(denominator) > 1e-10:
                        slope = numerator / denominator
                        intercept = y_mean - slope * x_mean
                    else:
                        slope = 0.0
                        intercept = y_mean
                    
                    # 约束系数
                    slope = np.clip(slope, -2.0, 2.0)
                    intercept = np.clip(intercept, min_val, max_val)
                    
                    # 检查端点约束
                    pred_at_min = intercept + slope * min_val
                    pred_at_max = intercept + slope * max_val
                    
                    if pred_at_min < min_val:
                        slope = (min_val - intercept) / (min_val + 1e-8)
                    if pred_at_max > max_val:
                        slope = (max_val - intercept) / (max_val + 1e-8)
                    
                    coeffs_fused[i, j] = [intercept, slope]
                else:
                    coeffs_fused[i, j, 0] = np.clip(combined_pred[i, j], min_val, max_val)
                    coeffs_fused[i, j, 1] = 0.0
        
        r = C_t2_band - combined_pred
        
        return coeffs_fused, r
    
    
    def select_similar_pixels_efficient(self):
        """
        高效的相似像素选择：向量化 + 预计算 + 优化权重
        """
        
        print("Selecting similar pixels efficiently...")

        h, w,_ = self.F_t1.shape
        similar_num = self.similar_num
        win_size = self.similar_win_size
        pad_size = win_size // 2
        
        # 获取分类结果
        F1_classified = self.classify_reference_images_fast_v2(self.F_t1, 5)
        
        # 填充
        F_t1_pad = np.pad(self.F_t1,
                        pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                        mode="reflect")
        F1_classified_pad = np.pad(F1_classified,
                                pad_width=((pad_size, pad_size), (pad_size, pad_size)),
                                mode="edge")
        
        # 预分配结果数组
        F_t1_similar_weights = np.zeros((h, w, similar_num), dtype=np.float32)
        F_t1_similar_indices = np.zeros((h, w, similar_num), dtype=np.int32)
        
        
        class_kd_trees = {}
        class_pixel_vals = {}  # 存储每个类别的像素值
        class_coords = {}      # 存储每个类别的坐标
        
        unique_classes = np.unique(F1_classified)
        print(f"Unique classes: {unique_classes}")

        
        for class_idx in unique_classes:
            rows, cols = np.where(F1_classified == class_idx)
            if len(rows) > 0:
                coords = np.column_stack([rows, cols])
                class_kd_trees[class_idx] = cKDTree(coords)
                class_coords[class_idx] = coords
                class_pixel_vals[class_idx] = self.F_t1[rows, cols, :]
                print(f"  Class {class_idx}: {len(rows)} pixels")
        
        print("Computing weights...")
        
        for row_idx in range(h):
            row_start = row_idx
            row_end = row_idx + win_size
            
            for col_idx in range(w):
                current_class = F1_classified[row_idx, col_idx]
                center_val = self.F_t1[row_idx, col_idx, :]
                
                # 获取窗口数据
                window_vals = F_t1_pad[row_start:row_end, 
                                    col_idx:col_idx + win_size, :]
                window_classes = F1_classified_pad[row_start:row_end,
                                                col_idx:col_idx + win_size]
                
                # 计算光谱差异            
                spectral_diffs = np.mean(np.abs(window_vals - center_val), axis=2)
                spectral_diffs_flat = spectral_diffs.ravel()
                
                # 同类像素掩码
                same_class_mask = (window_classes.ravel() == current_class)
                valid_indices = np.where(same_class_mask)[0]
                valid_diffs = spectral_diffs_flat[valid_indices]
                
                # 如果同类像素不足，从KD树补充
                if len(valid_indices) < similar_num and current_class in class_kd_trees:
                    kd_tree = class_kd_trees[current_class]
                    coords = class_coords[current_class]
                    pixel_vals = class_pixel_vals[current_class]
                    
                    # 查询最近的像素
                    k = min(len(coords), similar_num)
                    if k > 0:
                        dist, idx = kd_tree.query([[row_idx, col_idx]], k=k)
                        
                        # 逐个处理找到的像素
                        for i, neighbor_idx in enumerate(idx[0]):
                            if neighbor_idx >= len(coords):
                                continue
                                
                            nr, nc = coords[neighbor_idx]
                            nr, nc = int(nr), int(nc)
                            
                            # 跳过已经在窗口内的像素
                            if abs(nr - row_idx) <= pad_size and abs(nc - col_idx) <= pad_size:
                                continue
                            
                            # 计算局部窗口索引
                            local_r = nr - row_idx + pad_size
                            local_c = nc - col_idx + pad_size
                            
                            if 0 <= local_r < win_size and 0 <= local_c < win_size:
                                local_idx = local_r * win_size + local_c
                                
                                # 计算光谱差异
                                neighbor_val = pixel_vals[neighbor_idx]
                                spec_diff = np.mean(np.abs(neighbor_val - center_val))
                                
                                valid_indices = np.append(valid_indices, local_idx)
                                valid_diffs = np.append(valid_diffs, spec_diff)
                
                # 选择最相似的像素
                if len(valid_indices) > 0:
                    n_use = min(similar_num, len(valid_indices))
                    
                    # 按光谱差异排序
                    sorted_idx = np.argsort(valid_diffs)[:n_use]
                    use_indices = valid_indices[sorted_idx]
                    use_diffs = valid_diffs[sorted_idx]
                    
                    # 计算权重
                    if n_use > 1:
                        weights = 1.0 / (use_diffs+ 0.000001)
                        weights = weights / (np.sum(weights)  + 0.0000001)

                    else:
                        weights = np.ones(1)
                    
                    # 存储结果
                    F_t1_similar_indices[row_idx, col_idx, :n_use] = use_indices
                    F_t1_similar_weights[row_idx, col_idx, :n_use] = weights
                    
                    # 如果不足，用第一个索引填充
                    if n_use < similar_num:
                        F_t1_similar_indices[row_idx, col_idx, n_use:] = use_indices[0]
            
            if (row_idx + 1) % 20 == 0:
                print(f"  Progress: {(row_idx + 1) / h * 100:.1f}%")
        
        print("Finished selecting similar pixels!")
        return F_t1_similar_indices, F_t1_similar_weights, F1_classified


    def spatial_filtering_simple_enhanced(self, F_t2_RM_pred, F_t1_similar_indices, F_t1_similar_weights,band_idx):
        """
        空间增强滤波
        """
            
        h, w = F_t2_RM_pred.shape
        win_size = self.similar_win_size
        pad_size = win_size // 2
        F1_pad = self.F_t1[:,:,band_idx]
        
        print(f"  Processing {h}x{w} image with window size {win_size}...")
        
        # ============ 计算边缘强度 ============
        grad_x = sobel(F_t2_RM_pred, axis=0)
        grad_y = sobel(F_t2_RM_pred, axis=1)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        edge_strength_norm = edge_strength / (np.max(edge_strength) + 1e-8)
        
        # ============ 填充图像 ============
        F_t2_RM_pred_pad = np.pad(F_t2_RM_pred,
                                pad_width=((pad_size, pad_size), (pad_size, pad_size)),
                                mode="reflect")
        
        # ============ 初始化结果 ============
        SF_pred = np.zeros_like(F_t2_RM_pred)
        
        # ============ 主循环 ============
        for row_idx in range(h):
            for col_idx in range(w):
                # 获取窗口
                neighbor_pixel_RM_pred = F_t2_RM_pred_pad[row_idx:row_idx + win_size,
                                                        col_idx:col_idx + win_size]
                
                # 获取相似像素信息
                similar_indices = F_t1_similar_indices[row_idx, col_idx, :]
                similar_weights = F_t1_similar_weights[row_idx, col_idx, :]
                
                # 筛选有效像素
                valid_mask = similar_weights > 0
                
                if not np.any(valid_mask):
                    # 无相似像素，使用窗口内中值
                    SF_pred[row_idx, col_idx] = np.median(neighbor_pixel_RM_pred)
                    continue
                
                # 获取有效索引和权重
                valid_indices = similar_indices[valid_mask]
                valid_weights = similar_weights[valid_mask]
                # ============ 改进1：确保权重归一化 ============
                valid_weights = valid_weights / np.sum(valid_weights)
                
                # 展平窗口并获取对应的预测值
                window_flat = neighbor_pixel_RM_pred.flatten()
                similar_RM_pred = window_flat[valid_indices]
                
                # 原始计算方式：similar_weights * similar_RM_pred 的和
                original_result = np.sum(valid_weights * similar_RM_pred)
                
                # 边缘自适应：边缘区域保留更多原始值
                alpha = edge_strength_norm[row_idx, col_idx]
                SF_pred[row_idx, col_idx] = alpha * F1_pad[row_idx, col_idx] + \
                                          (1 - alpha) * original_result
            if (row_idx + 1) % 100 == 0:
                print(f"    Progress: {(row_idx + 1) / h * 100:.1f}%")
        
        print(f"  Finished processing")
        return SF_pred

    def residual_compensation_up(self, F_t2_SF_pred, residuals, 
                                            F_t1_similar_indices, 
                                            F_t1_similar_weights, band_idx):
        """
        简化的引导滤波差异图残差补偿
        """        
        
        h, w = residuals.shape
        win_size = self.similar_win_size
        pad_size = win_size // 2
        
        # ============ 步骤1: 计算初始差异图 ============
        C_Upscale = resize(self.C_t1[:,:,band_idx], (h, w), order=3, preserve_range=True)
        F_t1_band = self.F_t1[:, :, band_idx]     
        diff_map = np.abs(C_Upscale-F_t1_band)
        rmse = np.sqrt(np.mean(diff_map ** 2))
        dif_map_lr = diff_map*rmse
        
        # ============ 步骤2: 引导滤波优化 ============
        def simple_guided(I, p, r=1, eps=0.01):
            """简化的引导滤波"""
            window = 2 * r + 1
            N = uniform_filter(np.ones_like(I), window, mode='reflect')
            
            mean_I = uniform_filter(I, window, mode='reflect') / N
            mean_p = uniform_filter(p, window, mode='reflect') / N
            mean_Ip = uniform_filter(I * p, window, mode='reflect') / N
            mean_II = uniform_filter(I * I, window, mode='reflect') / N
            
            a = (mean_Ip - mean_I * mean_p) / (mean_II - mean_I * mean_I + eps)
            b = mean_p - a * mean_I
            
            mean_a = uniform_filter(a, window, mode='reflect') / N
            mean_b = uniform_filter(b, window, mode='reflect') / N
            
            return mean_a * I + mean_b
    
        
        # 对差异图应用引导滤波
        dif_map_lr_refined = simple_guided(F_t1_band, dif_map_lr, r=1, eps=0.01)
        
        # ============ 步骤4: 填充和补偿 ============
        residuals_pad = np.pad(residuals, pad_size, mode='reflect')
        dif_pad = np.pad(dif_map_lr_refined, pad_size, mode='reflect')
        
        Fit_FC_pred = F_t2_SF_pred.copy()
        pred_residuals = np.zeros_like(residuals)
        
        for i in range(h):
            for j in range(w):
                r_window = residuals_pad[i:i+win_size, j:j+win_size].ravel()
                d_window = dif_pad[i:i+win_size, j:j+win_size].ravel()
                
                idx = F_t1_similar_indices[i, j]
                wgt = F_t1_similar_weights[i, j]
                
                valid = wgt > 0
                if not np.any(valid):
                    continue
                
                idx_valid = idx[valid]
                wgt_valid = wgt[valid]
                
                valid_idx = idx_valid < len(r_window)
                if not np.any(valid_idx):
                    continue
                
                idx_final = idx_valid[valid_idx]
                wgt_final = wgt_valid[valid_idx]
                r_vals = r_window[idx_final]
                d_vals = d_window[idx_final]
                
                # 使用优化后的差异图计算权重
                d_weights = d_vals / (np.max(d_vals) + 1e-8)
                combined_weights = wgt_final * d_weights  
                residual = np.sum(r_vals * combined_weights)
                
                pred_residuals[i, j] = residual
                Fit_FC_pred[i, j] += residual
    
        return Fit_FC_pred, pred_residuals

    def mrtf(self):
        RM_pred = np.empty(shape=self.F_t1.shape, dtype=np.float32)
        SF_pred = np.empty(shape=self.F_t1.shape, dtype=np.float32)
        mrtf_pred = np.empty(shape=self.F_t1.shape, dtype=np.float32)
        h, w, c = self.F_t1.shape

        similar_indices, similar_weights, F1_classified = self.select_similar_pixels_efficient()
        print("Selected similar pixels!")
        use_combined = True
        for band_idx in range(self.F_t1.shape[2]):
            if use_combined == True:
                # 使用多尺度Huber回归
                coeffs, r = self.regression_model_fitting_combined(band_idx,non_linear_ratio=0.3, min_val = 0.0, max_val=1.0)
            else:
                coeffs, r = self.regression_model_fitting_combined_v2(band_idx,non_linear_ratio=0.3, min_val = 0, max_val=1.0)
    
            # 为了保持变量名一致，将一次项系数赋值给a，常数项赋值给b
            a = coeffs[:, :, 1]  # 一次项系数（斜率）
            b = coeffs[:, :, 0]  # 常数项（截距）
            
            # 调整尺寸
            a_resized = resize(a, output_shape=(self.F_t1.shape[0], self.F_t1.shape[1]), order=1, preserve_range=True)
            b_resized = resize(b, output_shape=(self.F_t1.shape[0], self.F_t1.shape[1]), order=1, preserve_range=True)
            r_resized = resize(r, output_shape=(self.F_t1.shape[0], self.F_t1.shape[1]), order=3, preserve_range=True)

            band_RM_pred = self.F_t1[:, :, band_idx] * a_resized + b_resized    
            print(f"Finished RM prediction of band {band_idx}!")

            band_SF_pred = self.spatial_filtering_simple_enhanced(band_RM_pred, similar_indices, similar_weights,band_idx)
            print(f"Finished spatial filtering of band {band_idx}!")

            band_mrtf_pred, pred_residuals = self.residual_compensation_up(band_SF_pred, r_resized, similar_indices,
                                                                          similar_weights,band_idx)
            print(f"Finished final prediction of band {band_idx}!")
            RM_pred[:, :, band_idx] = band_RM_pred
            SF_pred[:, :, band_idx] = band_SF_pred
            mrtf_pred[:, :, band_idx] = band_mrtf_pred

        return RM_pred, SF_pred, mrtf_pred


###########################################################
#                  Parameters setting                     #
###########################################################
scale_factor = 4
similar_win_size = 20
similar_num = 5

F_tb_path = r"C:\train\20211220_20221224\G_20211220_norl.tif"
C_tb_path = r"C:\train\20211220_20221224\L_20211220_norl.tif"
C_tp_path = r"C:\train\20211220_20221224\L_20221224_norl.tif"
mrtf_path = r"C:\train\20211220_20221224\L_20221224_norl_mrtf.tif"

if __name__ == "__main__":
    F_tb, F_tb_profile = read_raster(F_tb_path)
    C_tb = read_raster(C_tb_path)[0]
    C_tb_coarse = downscale_local_mean(C_tb, factors=(scale_factor, scale_factor, 1))
    C_tp = read_raster(C_tp_path)[0]
    C_tp_coarse = downscale_local_mean(C_tp, factors=(scale_factor, scale_factor, 1))

    time0 = datetime.now()
    proposed_mrtf = proposed(F_tb, C_tb_coarse, C_tp_coarse,
                    scale_factor=scale_factor,
                    similar_win_size=similar_win_size, similar_num=similar_num)

    F_tp_RM, F_tp_SF, F_tp_mrtf = proposed_mrtf.mrtf()
    time1 = datetime.now()
    time_span = time1 - time0
    print(f"Used {time_span.total_seconds():.2f} seconds!")

    write_raster(F_tp_mrtf, F_tb_profile, mrtf_path)
