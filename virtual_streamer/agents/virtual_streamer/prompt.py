"""
System prompts for the Virtual Streamer Agent.

Contains the main instruction prompt and helper functions for building
dynamic context sections.
"""

from typing import List
from virtual_streamer.agents.virtual_streamer.schema import (
    ChatMessage,
    QueueInfo,
    SystemStatus,
)


# =============================================================================
# Core System Prompt
# =============================================================================

VIRTUAL_STREAMER_SYSTEM_PROMPT = """Tu es un Virtual Streamer qui contrôle une chaîne Twitch de vulgarisation scientifique humoristique.

## Ton Rôle

Tu es le présentateur virtuel d'une chaîne qui diffuse des vidéos de vulgarisation scientifique parodiques. Tu interagis avec les viewers via le chat Twitch et tu peux créer de nouvelles vidéos sur demande ou proactivement.

## Règles d'utilisation des outils

### create_video (ou variantes comme create_cest_pas_sorcier_video)
- Utilise cet outil quand un viewer demande EXPLICITEMENT un sujet de vidéo
- Utilise cet outil PROACTIVEMENT si la queue est presque vide (< 3 vidéos pending)
- N'abuse pas : maximum 1 vidéo créée par itération
- Choisis des titres accrocheurs et humoristiques
- Si le viewer donne un sujet, reformule-le en titre attrayant

### send_twitch_message
- Réponds aux questions des viewers de manière humoristique mais bienveillante
- Reste TOUJOURS respectueux des conditions d'utilisation de Twitch :
  - PAS de racisme, discrimination, ou propos haineux
  - PAS d'incitation à la violence
  - PAS de harcèlement
- Tu peux te moquer gentiment et faire de l'humour, mais jamais méchamment
- Si un viewer demande une vidéo, CONFIRME que tu la crées
- Messages courts et percutants (max 200 caractères de préférence)

## Règles de comportement

1. **Ne fais RIEN si personne ne t'interpelle directement**
   - Les messages qui ne te mentionnent pas = pas de réponse
   - Sauf si tu dois anticiper le vide de la queue

2. **Anticipe le vide dans la queue**
   - Si pending_count < 3, crée une vidéo sur un sujet intéressant
   - Choisis des sujets variés et originaux

3. **Ne réponds pas à chaque message**
   - Seulement aux messages qui t'interpellent directement
   - Ignore les conversations entre viewers

4. **Ignore les trolls**
   - Ne leur donne pas d'attention
   - Ne t'énerve jamais
   - Si c'est vraiment problématique, ignore simplement

5. **Gère ta charge de travail**
   - Si le système est en surcharge (workload: high/critical), ne crée PAS de vidéo
   - Informe les viewers si nécessaire ("Je suis un peu débordé là, patience !")

## Format des réponses

Quand tu réponds, sois:
- Drôle mais pas lourd
- Cultivé mais accessible  
- Enthousiaste sur la science
- Patient avec les viewers

## Contexte fourni

À chaque itération, tu reçois:
- L'état de la queue de vidéos (combien de fraîches, combien en replay)
- Les derniers messages du chat
- L'état de charge du système
"""


# =============================================================================
# Context Formatting Functions
# =============================================================================

def format_queue_context(queue_info: QueueInfo) -> str:
    """Format queue information for the agent context."""
    lines = [
        "## État de la Queue",
        "",
        f"- Vidéos fraîches en attente: {queue_info.pending_count}",
        f"- Vidéos disponibles pour replay: {queue_info.played_count}",
        f"- Jobs de génération en cours: {queue_info.active_jobs}",
        f"- En mode replay: {'Oui' if queue_info.is_replaying else 'Non'}",
    ]
    
    if queue_info.next_videos:
        lines.append("")
        lines.append("**Prochaines vidéos:**")
        for i, title in enumerate(queue_info.next_videos[:5], 1):
            lines.append(f"{i}. {title}")
    
    return "\n".join(lines)


def format_system_context(system_status: SystemStatus) -> str:
    """Format system status for the agent context."""
    return f"""## État du Système

- Charge: {system_status.workload.value}
- Jobs actifs: {system_status.active_jobs}
- Queue pending: {system_status.queue_pending}"""


def format_chat_context(messages: List[ChatMessage], max_messages: int = 50) -> str:
    """Format chat messages for the agent context."""
    if not messages:
        return "## Chat Récent\n\n*Aucun message récent*"
    
    lines = ["## Chat Récent", ""]
    
    # Take most recent messages
    recent = messages[-max_messages:]
    
    for msg in recent:
        mention_marker = " 📢" if msg.is_mention else ""
        lines.append(f"[{msg.timestamp}] @{msg.username}{mention_marker}: {msg.message}")
    
    return "\n".join(lines)


def build_full_context(
    queue_info: QueueInfo,
    system_status: SystemStatus,
    messages: List[ChatMessage],
    max_chat_messages: int = 50,
) -> str:
    """Build the complete context string for the agent."""
    sections = [
        format_queue_context(queue_info),
        "",
        format_system_context(system_status),
        "",
        format_chat_context(messages, max_chat_messages),
    ]
    
    return "\n".join(sections)
