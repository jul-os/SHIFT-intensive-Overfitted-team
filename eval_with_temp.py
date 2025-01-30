import logging
import torch
import hydra
from omegaconf import OmegaConf
from srcs.utils import instantiate, get_logits
from srcs.postprocessing.temperature_scaling import train_temperature_scaling, apply_temperature_scaling


logger = logging.getLogger('Calculate calibration and classification metrics')


@hydra.main(version_base=None, config_path='conf', config_name='temperature')
def main(config):
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint, map_location=torch.device('cpu'))

    loaded_config = OmegaConf.create(checkpoint['config'])

    # setup data_loader instances
    data_loader = instantiate(config.data_loader)

    # restore network architecture
    model = instantiate(loaded_config.arch)

    # load trained weights
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # instantiate loss and metrics
    est_calibration_error = [instantiate(met, is_func=True) for met in config.metrics][0]

    logits, targets = get_logits(model, data_loader)

    temperature = train_temperature_scaling(logits, targets, config.temperature_scaling.initial_temperature)

    probs = apply_temperature_scaling(logits, temperature)
    ece_before_calibrate = est_calibration_error(logits.cpu(), targets.cpu())
    ece_after_calibrate = est_calibration_error(probs.cpu(), targets.cpu())

    logger.info(f'ECE before calibrating: {ece_before_calibrate}')
    logger.info(f'ECE after calibrating: {ece_after_calibrate}')
    logger.info(f'Temperature: {temperature}')

    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
