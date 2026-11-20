"""Record frames directly from the live MuJoCo pick-and-place simulation."""

from contextlib import contextmanager
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, FPS = 960, 540, 30


@contextmanager
def record_video(
    model: mujoco.MjModel, path: Path, *, caption: str = "", cube: int | None = None, fps: int = FPS,
):
    from imageio_ffmpeg import write_frames

    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.42, 0.02, 0.30)
    camera.distance = 1.44
    camera.azimuth = 126
    camera.elevation = -29
    title_font = ImageFont.load_default(size=22)
    detail_font = ImageFont.load_default(size=15)
    cube = model.geom("cube").id if cube is None else cube
    frame = 0

    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        writer = write_frames(
            str(path.resolve()), (WIDTH, HEIGHT), fps=fps, codec="libx264", quality=8,
            macro_block_size=2, output_params=["-movflags", "+faststart"],
        )
        writer.send(None)

        def capture(
            state: str, data: mujoco.MjData, left_n: float, right_n: float, *, active_cube: int | None = None,
        ) -> None:
            nonlocal frame
            if data.time + model.opt.timestep / 2 < frame / fps:
                return
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            panel_x = WIDTH - 395
            panel_bottom = 116 if caption else 94
            draw.rounded_rectangle((panel_x, 18, WIDTH - 18, panel_bottom), radius=10, fill="#162431")
            draw.text((panel_x + 13, 28), f"{state}    {data.time:05.2f} s", fill="#F2F7FA", font=title_font)
            height_mm = 1000 * data.geom_xpos[cube if active_cube is None else active_cube, 2]
            draw.text(
                (panel_x + 13, 62), f"Cube Z {height_mm:.1f} mm   |   Fingers {left_n:.2f} / {right_n:.2f} N",
                fill="#C3E7DF", font=detail_font,
            )
            if caption:
                draw.text((panel_x + 13, 87), caption, fill="#C3E7DF", font=detail_font)
            writer.send(image.tobytes())
            frame += 1

        try:
            yield capture
        finally:
            writer.close()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"MP4 encoding produced no video at {path}")
