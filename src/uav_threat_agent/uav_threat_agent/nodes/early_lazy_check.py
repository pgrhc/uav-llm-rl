import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LazyStrategyDetector(BaseCallback):
    """
    Her N timestep'te action istatistiklerini loglar.
    Lazy strategy (her şeye 0 verme) erkenden tespit edilir.
    
    ⚠️ ALARM KOŞULLARI:
      - mean_action < 0.15      → Agent çok az skor veriyor (lazy!)
      - action_std < 0.1        → Hiç çeşitlilik yok (deterministik lazy)
      - zero_ratio > 0.7        → Slot'ların %70'i 0 alıyor
    """
    
    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.actions_buffer = []
        
    def _on_step(self) -> bool:
        # Her step'te action'ı kaydet
        if len(self.locals["actions"].shape) == 2:
            # Batch olabilir, ilk örneği al
            action = self.locals["actions"][0]
        else:
            action = self.locals["actions"]
        
        self.actions_buffer.append(action.copy())
        
        # Check frequency'e ulaştı mı?
        if self.num_timesteps % self.check_freq == 0:
            self._analyze_and_report()
            self.actions_buffer = []  # Buffer'ı temizle
            
        return True
    
    def _analyze_and_report(self):
        if len(self.actions_buffer) == 0:
            return
        
        actions = np.array(self.actions_buffer)  # Shape: (N, 5)
        
        # ═══════════════════════════════════════════════════════════════
        # METRIC 1: Mean Action (Ortalama skor)
        # ═══════════════════════════════════════════════════════════════
        mean_action = float(np.mean(actions))
        
        # ═══════════════════════════════════════════════════════════════
        # METRIC 2: Action Std Dev (Çeşitlilik)
        # ═══════════════════════════════════════════════════════════════
        action_std = float(np.std(actions))
        
        # ═══════════════════════════════════════════════════════════════
        # METRIC 3: Zero Action Ratio (0'a yakın slot oranı)
        # ═══════════════════════════════════════════════════════════════
        zero_count = np.sum(actions < 0.1)
        zero_ratio = float(zero_count / actions.size)
        
        # ═══════════════════════════════════════════════════════════════
        # METRIC 4: High Action Ratio (Yüksek skor oranı)
        # ═══════════════════════════════════════════════════════════════
        high_count = np.sum(actions > 0.5)
        high_ratio = float(high_count / actions.size)
        
        # ═══════════════════════════════════════════════════════════════
        # METRIC 5: Action Distribution (Histogram)
        # ═══════════════════════════════════════════════════════════════
        bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
        hist, _ = np.histogram(actions, bins=bins)
        hist_str = " | ".join([f"{bins[i]}-{bins[i+1]}: {hist[i]}" 
                               for i in range(len(hist))])
        
        # ═══════════════════════════════════════════════════════════════
        # TensorBoard'a logla
        # ═══════════════════════════════════════════════════════════════
        self.logger.record("custom/mean_action", mean_action)
        self.logger.record("custom/action_std", action_std)
        self.logger.record("custom/zero_ratio", zero_ratio)
        self.logger.record("custom/high_ratio", high_ratio)
        
        # ═══════════════════════════════════════════════════════════════
        # ALARM KONTROLÜ
        # ═══════════════════════════════════════════════════════════════
        warnings = []
        
        if mean_action < 0.15:
            warnings.append("⚠️  LAZY STRATEGY: Mean action < 0.15")
        
        if action_std < 0.1:
            warnings.append("⚠️  NO DIVERSITY: Action std < 0.1")
        
        if zero_ratio > 0.7:
            warnings.append("⚠️  TOO MANY ZEROS: 70%+ slots get score ≈ 0")
        
        if high_ratio < 0.05:
            warnings.append("⚠️  NO HIGH SCORES: <5% slots get score > 0.5")
        
        # ═══════════════════════════════════════════════════════════════
        # KONSOL RAPORU
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print(f"📊 LAZY STRATEGY CHECK @ {self.num_timesteps:,} timesteps")
        print("="*70)
        print(f"  Mean Action:    {mean_action:.3f}")
        print(f"  Action Std:     {action_std:.3f}")
        print(f"  Zero Ratio:     {zero_ratio:.3f} ({zero_ratio*100:.1f}%)")
        print(f"  High Ratio:     {high_ratio:.3f} ({high_ratio*100:.1f}%)")
        print(f"\n  Distribution:   {hist_str}")
        
        if warnings:
            print("\n" + "🚨 ALARMLAR:")
            for w in warnings:
                print(f"  {w}")
            print("\n  → Reward fonksiyonunu kontrol et!")
            print("  → Person detection reward yeterince yüksek mi?")
            print("  → Critical miss penalty düşük threshold'da mı?")
        else:
            print("\n  ✅ Action distribution sağlıklı görünüyor.")
        
        print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# KULLANIM:
# ═══════════════════════════════════════════════════════════════════════════════
# train_debug_node.py içinde:
#
# from lazy_strategy_detector import LazyStrategyDetector
#
# lazy_detector = LazyStrategyDetector(
#     check_freq=10_000,  # Her 10k timestep'te kontrol et
#     verbose=1
# )
#
# callbacks = CallbackList([checkpoint_cb, lazy_detector])
#
# model.learn(..., callback=callbacks)
# ═══════════════════════════════════════════════════════════════════════════════