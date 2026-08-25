"""Minimal pygame Human-vs-DQN Space Invaders demo."""

import sys
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pygame
import torch
from huggingface_hub import hf_hub_download
from torch import no_grad

from src.models.cnn import CNNModelPY
from src.utils.preprocessor import FrameSkipStepper, Preprocessor
from src.utils.space_invaders import make_space_invaders_env


# Change these two values when publishing a replacement trained checkpoint.
HF_REPO_ID = "colichar/space-invaders-dqn"
HF_CHECKPOINT_FILENAME = "model.pth"
EMULATOR_FPS = 60
SCREEN_SIZE = (320, 420)  # Atari's 160x210 screen, scaled 2x.


def load_model(path, n_actions, device):
    """Load the model weights from the project's standard ``model.pth`` checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = CNNModelPY(n_actions, device=device)
    model.set_weights(checkpoint["model_state_dict"])
    model.eval()
    return model


def action_lookup(env):
    """Return ALE actions by name, accommodating ALE's action-space variants."""
    meanings = env.unwrapped.get_action_meanings()
    return {name.upper(): index for index, name in enumerate(meanings)}


def human_action(actions, keys):
    """Map held keys to ALE's combined movement/fire actions where available."""
    left = keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]
    right = keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]
    fire = keys[pygame.K_SPACE]
    direction = "LEFT" if left else "RIGHT" if right else ""
    name = f"{direction}FIRE" if fire and direction else (direction or ("FIRE" if fire else "NOOP"))
    # Space Invaders exposes LEFTFIRE/RIGHTFIRE. If a future ALE variant does
    # not, favor movement over firing rather than sending an invalid action.
    return actions.get(name, actions.get(direction, actions.get("FIRE", actions["NOOP"])))


class Match:
    """Owns two independent environments and their small amount of match state."""

    def __init__(self, model):
        self.model = model
        self.preprocessor = Preprocessor()
        self.human_env = make_space_invaders_env()
        self.agent_env = make_space_invaders_env()
        self.human_actions = action_lookup(self.human_env)
        self.reset()

    def reset(self):
        self.human_frame, human_info = self.human_env.reset()
        self.agent_state, agent_info = self.preprocessor.initialize_state(self.agent_env)
        self.agent_frame = self.agent_env.render()
        self.human_score = self.agent_score = 0.0
        self.human_lives = human_info.get("lives")
        self.agent_lives = agent_info.get("lives")
        self.human_done = self.agent_done = False
        self.agent_stepper = None

    @staticmethod
    def _finished(terminated, truncated, info):
        return terminated or truncated or info.get("lives") == 0

    def step_human(self, keys):
        if self.human_done:
            return
        self.human_frame, reward, terminated, truncated, info = self.human_env.step(
            human_action(self.human_actions, keys)
        )
        self.human_score += reward
        self.human_lives = info.get("lives", self.human_lives)
        self.human_done = self._finished(terminated, truncated, info)

    def step_agent(self):
        if self.agent_done:
            return
        if self.agent_stepper is None:
            with no_grad():
                action = self.model.best_action(
                    self.agent_state.unsqueeze(0).to(self.model.device).float().div(255.0)
                ).item()
            self.agent_stepper = FrameSkipStepper(self.agent_env, action, self.preprocessor.frame_skip)

        self.agent_frame, _, _, _, _ = self.agent_stepper.advance()
        if self.agent_stepper.complete:
            raw_obs, reward, terminated, truncated, info = self.agent_stepper.result()
            self.agent_score += reward
            self.agent_lives = info.get("lives", self.agent_lives)
            self.agent_state, _ = self.preprocessor.new_state(raw_obs, self.agent_state)
            self.agent_done = self._finished(terminated, truncated, info)
            self.agent_stepper = None

    @property
    def complete(self):
        return self.human_done and self.agent_done

    def close(self):
        self.human_env.close()
        self.agent_env.close()


def _surface(frame):
    # pygame expects (width, height, channels); ALE supplies (height, width, channels).
    surface = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
    return pygame.transform.scale(surface, SCREEN_SIZE)


def _draw_text(screen, font, text, center, color=(235, 235, 235)):
    screen.blit(font.render(text, True, color), font.render(text, True, color).get_rect(center=center))


def draw(screen, match, title_font, text_font):
    screen.fill((18, 18, 24))
    width = screen.get_width()
    _draw_text(screen, title_font, "SPACE INVADERS — HUMAN vs DQN", (width // 2, 25))
    panels = (("YOU", match.human_frame, match.human_score, match.human_lives, match.human_done, 10),
              ("DQN AGENT", match.agent_frame, match.agent_score, match.agent_lives, match.agent_done, 350))
    for label, frame, score, lives, done, x in panels:
        _draw_text(screen, text_font, label, (x + SCREEN_SIZE[0] // 2, 58))
        screen.blit(_surface(frame), (x, 75))
        _draw_text(screen, text_font, f"Score: {int(score)}", (x + 160, 515))
        if lives is not None:
            _draw_text(screen, text_font, f"Lives: {lives}", (x + 160, 540))
        if done:
            _draw_text(screen, text_font, "GAME OVER", (x + 160, 285), (255, 190, 80))
    if match.complete:
        if match.human_score == match.agent_score:
            outcome = "TIE"
        elif match.human_score > match.agent_score:
            outcome = "YOU WIN"
        else:
            outcome = "DQN WINS"
        _draw_text(screen, title_font, outcome, (width // 2, 574), (255, 210, 90))
    _draw_text(screen, text_font, "Left / Right Move    Space Fire    R Restart    Esc Quit", (width // 2, 620))
    pygame.display.flip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    probe_env = make_space_invaders_env()
    try:
        print("Downloading trained DQN model (or loading it from the local cache)...")
        checkpoint = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CHECKPOINT_FILENAME)
        model = load_model(checkpoint, probe_env.action_space.n, device)
    finally:
        probe_env.close()

    pygame.init()
    screen = pygame.display.set_mode((680, 650))
    pygame.display.set_caption("Space Invaders — Human vs DQN")
    match = Match(model)
    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 28)
    text_font = pygame.font.Font(None, 24)
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    match.reset()
            if running:
                keys = pygame.key.get_pressed()
                if not match.complete:
                    match.step_human(keys)
                    match.step_agent()
                draw(screen, match, title_font, text_font)
                clock.tick(EMULATOR_FPS)
    finally:
        match.close()
        pygame.quit()


if __name__ == "__main__":
    main()
