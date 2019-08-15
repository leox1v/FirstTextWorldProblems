from ipywidgets import widgets
import IPython
from IPython.display import display
from utils import bcolors

class Output:
    def __init__(self, init_text=None):
        self.out = widgets.Output(layout={'border': '1px solid black'})
        self.text = ''
        if init_text is not None:
            self.update_text(init_text)
            self.refresh_output()

    def update_text(self, text, append=False):
        if append:
            self.text += text
        else:
            self.text = text
        return text

    def refresh_output(self):
        self.out.clear_output()
        with self.out:
            print(self.text)


class OutputHandler:
    def __init__(self):
        env_out = Output()
        cmd_out = Output()
        rec_out = Output("Didn't examine the cookbook yet!")

        self.accordion = widgets.Accordion(children=[env_out.out, rec_out.out])
        self.accordion.set_title(0, bcolors.WORLD + ' Environment Output')
        # self.accordion.set_title(1, bcolors.ROBOT + ' Predicted Commands')
        self.accordion.set_title(1, bcolors.CHEF + ' Recipe')

        self.out = {
            'env': env_out,
            'cmd': cmd_out,
            'rec': rec_out
        }

    def display(self):
        display(self.accordion)

    def reset(self):
        IPython.display.clear_output()

    def refresh(self):
        self.reset()
        self.display()

    def update(self, text, box, append=False):
        """
        :param text: new text that goes into the box.
        :param box: string identifier of the output box
        :param append: Bool if text is appended or replaced
        """
        out = self.out[box]
        text = out.update_text(text, append=append)
        out.refresh_output()

class InputHandler:
    def __init__(self, callback):
        self.input = widgets.Text(
                            value='',
                            placeholder='Next Command',
                            description=bcolors.HUMAN,
                            disabled=False,
                            continuous_update=False
                        )

        self.text = ''
        self.box = None

        self.input.continuous_update = False
        self.input.observe(self.handle_submit, 'value')
        self.callback = callback
        self.ignore_next_change = False

    def handle_submit(self, sender):
        if self.ignore_next_change:
            self.ignore_next_change = False
            return
        self.text = self.input.value
        self.callback(self.text)
        self.ignore_next_change = True
        self.input.value = ''
        self.input.placeholder = 'Next command'

    def hide(self):
        self.input.layout.display = 'none'

    def display(self, add_out=None):
        if add_out is not None:
            label = widgets.Label('  '+ bcolors.CHEF + "Le DeepChef's thoughts:" )
            self.box = widgets.HBox([self.input, label, add_out])
            display(self.box)
        else:
            display(self.input)

    def reset(self):
        IPython.display.clear_output()

    def refresh(self):
        self.reset()
        self.display()




