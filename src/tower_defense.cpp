#include <tower_defense.h>

void
controller_reset(game_controller *G)
{
    G->Up = 0;
    G->Down = 0;
    G->Left = 0;
    G->Right = 0;
};

void
move_entity_0(entity_megastruct *Entity)
{
    Entity->Movement.Direction.z = -Entity->Controller->Left;
    Entity->Movement.Direction.x = Entity->Controller->Up;

    Entity->Movement.Position += Entity->Movement.Direction * glm::vec3(Entity->Movement.Speed, 0, Entity->Movement.Speed);
};