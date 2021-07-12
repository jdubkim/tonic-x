import gin
import tonic


@gin.configurable
def get_agent(agent=None, name=None):
    ''' Returns an agent from gin configuration.'''

    # Returns a default agent if agent was not specified
    if agent is None:
        agent = tonic.agents.NormalRandom()

    if not name:
        if hasattr(agent, 'name'):
            name = agent.name
        else:
            name = agent.__class__.__name__

    return agent, name
