"""Independently trained TCN event classifier with reference-interval pooling."""
from teacher_f import FrameTeacher


class EventTeacher(FrameTeacher):
    def forward(self, x, pool_mask):
        if pool_mask.shape != (x.shape[0],x.shape[-1]) or not pool_mask.any(dim=1).all():
            raise ValueError("Every event needs at least one pooling frame")
        h = self.encode_features(x)
        mask = pool_mask.unsqueeze(1).to(h.dtype)
        pooled = (h*mask).sum(dim=2,keepdim=True)/mask.sum(dim=2,keepdim=True)
        return [head(pooled) for head in self.heads]
