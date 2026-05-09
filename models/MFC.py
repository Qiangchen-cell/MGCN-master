import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryCenter(nn.Module):
    def __init__(self, feat_dim, memory_size=100):
        super().__init__()
        self.memory_size = memory_size
        self.feat_dim = feat_dim

        # 记忆存储
        self.memory = nn.Parameter(torch.zeros(memory_size, feat_dim))
        # 状态跟踪
        self.register_buffer('usage_count', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('is_filled', torch.zeros(memory_size, dtype=torch.bool))
        self.register_buffer('next_empty_slot', torch.tensor(0, dtype=torch.long))
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

        # 超参数
        self.momentum = 0.95
        self.similarity_threshold = 0.8

    def update_memory(self, features):
        """统一的记忆更新"""
        features = F.normalize(features, dim=-1)

        # 检查是否有空槽
        empty_indices = torch.where(~self.is_filled)[0]
        has_empty_slots = empty_indices.numel() > 0

        if has_empty_slots:
            self._fill_empty_slots(features)
        else:
            self._update_existing_slots(features)

        # 定期内存管理
        if self.step % 200 == 0 and self.step > 1000:
            self._manage_memory()

        self.step += 1

    def _fill_empty_slots(self, features):
        """填充空记忆槽"""
        batch_size = features.size(0)

        # 找到所有空槽
        empty_indices = torch.where(~self.is_filled)[0]
        num_empty = empty_indices.numel()

        if num_empty == 0:
            return

        num_to_fill = min(batch_size, num_empty)

        # 填充空槽
        for i in range(num_to_fill):
            idx = empty_indices[i]
            self.memory.data[idx] = features[i]
            self.is_filled[idx] = True
            self.usage_count[idx] = 1

        # 更新next_empty_slot
        remaining_empty = torch.where(~self.is_filled)[0]
        if remaining_empty.numel() > 0:
            self.next_empty_slot = remaining_empty.min()
        else:
            self.next_empty_slot = torch.tensor(self.memory_size, dtype=torch.long, device=self.memory.device)

    def _update_existing_slots(self, features):
        """更新现有的记忆槽"""
        filled_indices = torch.where(self.is_filled)[0]
        if filled_indices.numel() == 0:
            # 没有已填充的槽位，尝试填充空槽
            self._fill_empty_slots(features)
            return

        # 只考虑已填充的记忆槽进行相似度计算
        valid_memory = self.memory[filled_indices]
        memory_norm = F.normalize(valid_memory, dim=1)
        features_norm = F.normalize(features, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(features_norm, memory_norm.T)

        for i in range(features.size(0)):
            max_sim, max_idx_in_filled = torch.max(sim_matrix[i], dim=0)
            target_idx = filled_indices[max_idx_in_filled]

            if max_sim > self.similarity_threshold:
                # 动量更新：与最相似的记忆槽融合
                new_feat = self.momentum * self.memory.data[target_idx] + (1 - self.momentum) * features[i]
                self.memory.data[target_idx] = F.normalize(new_feat.unsqueeze(0), dim=-1).squeeze(0)
                self.usage_count[target_idx] += 1
                self.is_filled[target_idx] = True
            else:
                filled_usage = self.usage_count[filled_indices]
                min_usage_idx_in_filled = torch.argmin(filled_usage)
                replace_idx = filled_indices[min_usage_idx_in_filled]
                self.memory.data[replace_idx] = features[i]
                self.usage_count[replace_idx] = 1
                self.is_filled[replace_idx] = True

    def _manage_memory(self):
        """内存管理：检测并合并重复记忆槽"""
        filled_indices = torch.where(self.is_filled)[0]
        if filled_indices.numel() < 5:
            return

        valid_memory = self.memory[filled_indices]
        memory_norm = F.normalize(valid_memory, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(memory_norm, memory_norm.T)
        diag_mask = torch.eye(filled_indices.numel(), device=memory_norm.device).bool()
        sim_matrix_masked = sim_matrix.masked_fill(diag_mask, 0)

        # 找出高相似度的槽位
        high_sim_mask = sim_matrix_masked > 0.8
        high_sim_counts = high_sim_mask.sum(dim=1)

        redundant_indices_in_filled = torch.where(high_sim_counts >= 5)[0]
        if redundant_indices_in_filled.numel() == 0:
            return

        # 转换为原始索引
        redundant_slots = filled_indices[redundant_indices_in_filled]
        rep_slot_idx = redundant_slots[0]
        rep_idx_in_filled = (filled_indices == rep_slot_idx).nonzero(as_tuple=True)[0][0]

        similar_indices_in_filled = torch.where(high_sim_mask[rep_idx_in_filled])[0]
        similar_slots = filled_indices[similar_indices_in_filled]

        k = min(5, similar_slots.numel())
        if k == 0:
            return

        if similar_slots.numel() > k:
            sim_values = sim_matrix_masked[rep_idx_in_filled, similar_indices_in_filled]
            topk_sim, topk_idx = torch.topk(sim_values, k=k)
            similar_slots = similar_slots[topk_idx]

        # 记录需要清空的槽位
        slots_to_clear = []

        for sim_slot_idx in similar_slots:
            new_feat = self.momentum * self.memory.data[rep_slot_idx] + (1 - self.momentum) * self.memory.data[
                sim_slot_idx]
            self.memory.data[rep_slot_idx] = F.normalize(new_feat.unsqueeze(0), dim=-1).squeeze(0)
            self.usage_count[rep_slot_idx] += self.usage_count[sim_slot_idx]
            slots_to_clear.append(sim_slot_idx)

        # 清空标记的槽位
        for sim_slot_idx in slots_to_clear:
            self.memory.data[sim_slot_idx] = torch.zeros_like(self.memory.data[sim_slot_idx])
            self.usage_count[sim_slot_idx] = 0
            self.is_filled[sim_slot_idx] = False

        # 压缩内存
        self._compact_memory()

    def _compact_memory(self):
        """压缩记忆槽：将所有已填充的记忆槽移到前面"""
        valid_indices = torch.where(self.is_filled)[0]

        if valid_indices.numel() == 0:
            # 没有已填充的槽位，重置状态
            self.next_empty_slot = 0
            return

        new_next_empty = valid_indices.numel()

        # 移动已填充槽位的数据到前面
        if valid_indices.numel() > 0:
            valid_memory = self.memory[valid_indices].clone()
            valid_usage = self.usage_count[valid_indices].clone()

            self.memory.data[:new_next_empty] = valid_memory
            self.usage_count[:new_next_empty] = valid_usage

        # 清空后面的位置
        if new_next_empty < self.memory_size:
            self.memory.data[new_next_empty:] = 0
            self.usage_count[new_next_empty:] = 0

        # 更新is_filled状态
        self.is_filled[:] = False
        self.is_filled[:new_next_empty] = True

        # 更新next_empty_slot为第一个空槽的位置
        self.next_empty_slot = torch.tensor(new_next_empty, dtype=torch.long, device=self.memory.device)

    def retrieve(self, query, k=5):
        """改进的检索策略，考虑使用均衡"""
        query = F.normalize(query, dim=-1)

        filled_indices = torch.where(self.is_filled)[0]
        if filled_indices.numel() == 0:
            return query

        valid_memory = self.memory[filled_indices]
        memory_norm = F.normalize(valid_memory, dim=1)

        # 计算相似度
        sim = torch.matmul(query, memory_norm.T)

        k_actual = min(k, filled_indices.numel())
        topk_sim, topk_idx_in_filled = torch.topk(sim, k=k_actual, dim=-1)

        # 转换回原始索引
        topk_idx = filled_indices[topk_idx_in_filled]

        # 加权合并
        weights = F.softmax(topk_sim / 0.1, dim=-1)
        weights_expanded = weights.unsqueeze(-1)
        memory_selected = self.memory[topk_idx]
        retrieved = (memory_selected * weights_expanded).sum(dim=1)

        return F.normalize(retrieved, dim=-1)

class CrossModalAttention(nn.Module):
    def __init__(self, feat_dim=768, num_heads=16, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        self.cross_attn1 = self._create_attention_layer(feat_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(feat_dim)

        self.self_attn = self._create_attention_layer(feat_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(feat_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(feat_dim)

        self.output_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )

    def _create_attention_layer(self, feat_dim, num_heads, dropout):
        """创建标准的多头注意力层"""
        return nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, query_feat, key_feat, value_feat):
        query_seq = query_feat.unsqueeze(1)  # (batch, 1, feat_dim)
        key_seq = key_feat.unsqueeze(1)      # (batch, 1, feat_dim)
        value_seq = value_feat.unsqueeze(1)  # (batch, 1, feat_dim)

        residual1 = query_seq
        attn_output1, _ = self.cross_attn1(query_seq, key_seq, value_seq)
        output1 = self.norm1(residual1 + attn_output1)

        residual2 = output1
        attn_output2, _ = self.self_attn(output1, output1, output1)
        output2 = self.norm2(residual2 + attn_output2)

        residual3 = output2
        ff_output = self.ffn(output2)
        output3 = self.norm3(residual3 + ff_output)

        original_seq = query_feat.unsqueeze(1)
        combined = torch.cat([original_seq, output3], dim=-1)
        gate_weight = self.output_gate(combined)

        final_output = gate_weight * original_seq + (1 - gate_weight) * output3

        return final_output.squeeze(1)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(256)
        self.max_pool = nn.AdaptiveMaxPool1d(256)

        self.mlp = nn.Sequential(
            nn.Linear(256, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.mlp(avg_out)

        max_out = self.max_pool(x)
        max_out = self.mlp(max_out)

        channel_att = torch.sigmoid(avg_out + max_out) * x

        return channel_att

class TokenFeatureAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(TokenFeatureAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        b, d = x.size()

        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, num_heads]
        attn = attn.softmax(dim=-1)

        x_attn = (attn @ v).transpose(1, 2).reshape(b, d)  # [B, dim]
        x_attn = self.proj(x_attn) * x

        return x_attn

class FeatureRefiner(nn.Module):
    def __init__(self, dim, reduction_ratio=16):
        super(FeatureRefiner, self).__init__()
        self.dim = dim
        self.channel_att = ChannelAttention(dim, reduction_ratio)
        self.token_att = TokenFeatureAttention(dim)

        self.cross_attention_cond = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.cross_attention_mem = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.fusion_gate = nn.Sequential(nn.Linear(dim * 4, dim), nn.Sigmoid())

        self.use_memory = True
    def forward(self, x, cond_feat, mem_feat):
        original_x = x
        x = self.channel_att(x)
        x = self.token_att(x)

        cond_feat = self.channel_att(cond_feat)
        cond_feat = self.token_att(cond_feat)

        query = original_x.unsqueeze(1)
        key = value = cond_feat.unsqueeze(1)
        attended_features, _ = self.cross_attention_cond(query, key, value)
        cond_feat_x = attended_features.squeeze(1)

        if self.use_memory:
            mem_feat = self.channel_att(mem_feat)
            mem_feat = self.token_att(mem_feat)

            query = original_x.unsqueeze(1)
            key = value = mem_feat.unsqueeze(1)
            attended_features, _ = self.cross_attention_mem(query, key, value)
            mem_feat_x = attended_features.squeeze(1)

            fusion_input = torch.cat([original_x, x, cond_feat_x, mem_feat_x], dim=-1)

            fusion_weights = self.fusion_gate(fusion_input)
            enhanced_x = original_x + fusion_weights * (x + cond_feat_x + mem_feat_x)
        else:
            fusion_input_m = torch.cat([original_x, x, cond_feat_x,cond_feat_x], dim=-1)

            fusion_weights_m = self.fusion_gate(fusion_input_m)
            enhanced_x = original_x + fusion_weights_m * (x + cond_feat_x)

        return enhanced_x

    def gradient_boost_loss(self, original_features, enhanced_features, condition_features, memory_features):
        cond_alignment = F.mse_loss(enhanced_features, condition_features)
        improvement = F.mse_loss(enhanced_features, original_features)
        if self.use_memory:
            mem_guidance = F.mse_loss(enhanced_features, memory_features)
            total_loss = cond_alignment + mem_guidance + improvement
        else:
            total_loss = cond_alignment + improvement
        return total_loss

class FeatureGenerator(nn.Module):
    def __init__(self, image_feat_dim=768, text_feat_dim=768, noise_dim=100, hidden_dim=512):
        super(FeatureGenerator, self).__init__()

        self.image_feat_dim = image_feat_dim 
        self.text_feat_dim = text_feat_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim

        self.use_memory = True
        self.use_refine = True
        if self.use_memory:
            self.image_memory_center = MemoryCenter(feat_dim=image_feat_dim, memory_size=30)
            self.text_memory_center = MemoryCenter( feat_dim=text_feat_dim, memory_size=30)
        if self.use_refine:
            self.image_refiner = FeatureRefiner(dim=image_feat_dim, reduction_ratio=8)
            self.text_refiner = FeatureRefiner(dim=text_feat_dim, reduction_ratio=8)

        # 图像特征生成器（文本->图像）
        self.image_generator = nn.Sequential(
            nn.Linear(text_feat_dim + noise_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, image_feat_dim),
            nn.Tanh()
        )

        # 文本特征生成器（图像->文本）
        self.text_generator = nn.Sequential(
            nn.Linear(image_feat_dim + noise_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, text_feat_dim),
            nn.Tanh()
        )

        # 模态判别器
        self.modality_discriminator = nn.Sequential(
            nn.Linear(image_feat_dim + text_feat_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 模态一致性损失
        self.adversarial_loss = nn.BCELoss()
        self.consistency_loss = nn.CosineEmbeddingLoss()

        self.image_fusion_module = CrossModalAttention(feat_dim=text_feat_dim, num_heads=8, dropout=0.1)
        self.text_fusion_module = CrossModalAttention(feat_dim=image_feat_dim, num_heads=8, dropout=0.1)

        self.noise_modulation = nn.Sequential(
            nn.Linear(text_feat_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, noise_dim * 2),
            nn.Tanh() # 输出scale和bias
        )

    def generate_image_features(self, text_features, memory_features, noise):
        if self.use_memory:
            fused_condition = self.image_fusion_module(query_feat=memory_features, key_feat=text_features, value_feat=text_features)
            modulation_params = self.noise_modulation(fused_condition)
        else:
            fused_condition = text_features
            modulation_params = self.noise_modulation(fused_condition)

        scale, bias = torch.chunk(modulation_params, 2, dim=1)
        modulated_noise = noise * (1 + scale.tanh()) + bias

        combined = torch.cat([fused_condition, modulated_noise], dim=1)
        generated_images = self.image_generator(combined)

        return generated_images

    def generate_text_features(self, image_features, memory_features, noise):
        if self.use_memory:
            fused_condition = self.text_fusion_module(query_feat=memory_features, key_feat=image_features, value_feat=image_features)
            modulation_params = self.noise_modulation(fused_condition)
        else:
            fused_condition = image_features
            modulation_params = self.noise_modulation(fused_condition)

        scale, bias = torch.chunk(modulation_params, 2, dim=1)
        modulated_noise = noise * (1 + scale.tanh()) + bias

        combined = torch.cat([fused_condition, modulated_noise], dim=1)
        generated_texts = self.text_generator(combined)

        return generated_texts

    def discriminate_modality(self, image_features, text_features):
        """判别模态是否匹配"""
        combined = torch.cat([image_features, text_features], dim=1)
        validity = self.modality_discriminator(combined)
        return validity

    def compute_adversarial_loss(self, generated_features, condition_features):
        batch_size = condition_features.size(0)
        real_labels = torch.ones(batch_size, 1).to(condition_features.device)
        fake_labels = torch.zeros(batch_size, 1).to(condition_features.device)

        real_validity = self.discriminate_modality(condition_features, condition_features)
        fake_validity = self.discriminate_modality(generated_features, condition_features)

        d_loss_real = self.adversarial_loss(real_validity, real_labels)
        d_loss_fake = self.adversarial_loss(fake_validity, fake_labels)
        d_loss = (d_loss_fake + d_loss_real) / 2

        g_loss = self.adversarial_loss(fake_validity, real_labels)

        return d_loss + g_loss

    def compute_consistency_loss(self, feat1, feat2):
        """计算模态一致性损失"""
        target = torch.ones(feat1.size(0)).to(feat1.device)
        return self.consistency_loss(feat1, feat2, target)


    def forward(self, image_features, text_features, image_missing, text_missing):

        completed_image_features = image_features.clone()
        completed_text_features = text_features.clone()
        loss_gan = torch.tensor(0.0, device=image_features.device)
        loss_refiner = torch.tensor(0.0, device=image_features.device)

        complete_mask = ~(image_missing | text_missing)
        if self.use_memory and complete_mask.any():
            complete_images = image_features[complete_mask]
            complete_texts = text_features[complete_mask]

            # 更新记忆中心
            self.image_memory_center.update_memory(complete_images)
            self.text_memory_center.update_memory(complete_texts)

        # 处理图像缺失的样本
        image_missing_mask = image_missing
        if image_missing_mask.sum() > 0:
            missing_indices = image_missing_mask.nonzero(as_tuple=True)[0]
            condition_texts = text_features[missing_indices]

            if self.use_memory:
                image_memory_features = self.image_memory_center.retrieve(condition_texts, k=5)
            else:
                image_memory_features = None

            # 生成图像特征
            noise = torch.randn(len(missing_indices), self.noise_dim).to(image_features.device)
            generated_images = self.generate_image_features(condition_texts, image_memory_features, noise)
            generated_images_gan = generated_images.clone()

            if self.use_memory:
                loss_gan += self.compute_adversarial_loss(generated_images, image_memory_features)
            else:
                loss_gan += self.compute_adversarial_loss(generated_images, condition_texts)
            loss_gan += self.compute_consistency_loss(generated_images, condition_texts)

            if self.use_refine:
                generated_images = self.image_refiner(generated_images, condition_texts, image_memory_features)
                generated_images_refine = generated_images.clone()
                loss_refiner += self.image_refiner.gradient_boost_loss(generated_images_gan, generated_images_refine, condition_texts, image_memory_features)
            completed_image_features[missing_indices] = generated_images

            if self.use_memory:
                self.image_memory_center.update_memory(generated_images)

        # 处理文本缺失的样本
        text_missing_mask = text_missing
        if text_missing_mask.sum() > 0:
            missing_indices = text_missing_mask.nonzero(as_tuple=True)[0]
            condition_images = image_features[missing_indices]

            if self.use_memory:
                text_memory_features = self.text_memory_center.retrieve(condition_images, k=5)
            else:
                text_memory_features = None

            # 生成文本特征
            noise = torch.randn(len(missing_indices), self.noise_dim).to(image_features.device)
            generated_texts = self.generate_text_features(condition_images, text_memory_features, noise)
            generated_texts_gan = generated_texts.clone()

            if self.use_memory:
                loss_gan += self.compute_adversarial_loss(generated_texts, text_memory_features)
            else:
                loss_gan += self.compute_adversarial_loss(generated_texts, condition_images)
            loss_gan += self.compute_consistency_loss(generated_texts, condition_images)

            if self.use_refine:
                generated_texts = self.text_refiner(generated_texts, condition_images, text_memory_features)
                generated_texts_refine = generated_texts.clone()
                loss_refiner += self.text_refiner.gradient_boost_loss(generated_texts_gan, generated_texts_refine, condition_images, text_memory_features)
            completed_text_features[missing_indices] = generated_texts

            if self.use_memory:
                self.text_memory_center.update_memory(generated_texts)

        loss_mfc = loss_gan * 0.1 + loss_refiner * 1
        return completed_image_features, completed_text_features, loss_mfc

