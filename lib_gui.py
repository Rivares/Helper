# coding: UTF-8

from lib_gui_table import Table, TableView, TableColumn, TableRow

from matplotlib import pyplot as plt
import numpy as np

from kivy_garden.graph import Graph, MeshLinePlot

from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, StringProperty
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.label import Label
from kivy.graphics import Color
from kivy.vector import Vector
from kivy.config import Config
from kivy.clock import Clock
from kivy.app import App


Config.set("input", "mouse", "mouse, disable_multitouch")


red = [1, 0, 0, 1]
green = [0, 1, 0, 1]
blue = [0, 0, 1, 1]
purple = [1, 0, 1, 1]


class MainScreen(BoxLayout):
    """docstring for MainScreen"""
    def __init__(self):
        super(MainScreen, self).__init__()
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.my_table = Table()
        self.my_table.cols = 2
        self.my_table.add_button_row('123', '456')
        for i in range(110):
            self.my_table.add_row([Button, {'text': 'button%s' % i,
                                            'color_widget': [0, 0, 0.5, 1],
                                            'color_click': [0, 1, 0, 1]
                                            }],
                                  [TextInput, {'text': 'textinput%s' % i,
                                               'color_click': [1, 0, .5, 1]
                                               }])
        self.my_table.label_panel.visible = False
        self.my_table.label_panel.height_widget = 50
        self.my_table.number_panel.auto_width = False
        self.my_table.number_panel.width_widget = 100
        self.my_table.number_panel.visible = False
        self.my_table.choose_row(3)
        self.my_table.del_row(5)
        self.my_table.grid.color = [1, 0, 0, 1]
        self.my_table.label_panel.color = [0, 1, 0, 1]
        self.my_table.number_panel.color = [0, 0, 1, 1]
        self.my_table.scroll_view.bar_width = 10
        self.my_table.scroll_view.scroll_type = ['bars']
        self.my_table.grid.cells[0][0].text = 'edited button text'
        self.my_table.grid.cells[1][1].text = 'edited textinput text'
        self.my_table.grid.cells[3][0].height = 100
        self.my_table.label_panel.labels[1].text = 'New name'
        print("ROW COUNT:", self.my_table.row_count)
        self.add_widget(self.my_table)

    def _keyboard_closed(self):
        pass

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        """ Method of pressing keyboard  """
        if keycode[0] == 273:   # UP
            print(keycode)
            self.my_table.scroll_view.up()
        if keycode[0] == 274:   # DOWN
            print(keycode)
            self.my_table.scroll_view.down()
        if keycode[0] == 281:   # PageDown
            print(keycode)
            self.my_table.scroll_view.pgdn()
        if keycode[0] == 280:   # PageUp
            print(keycode)
            self.my_table.scroll_view.pgup()
        if keycode[0] == 278:   # Home
            print(keycode)
            self.my_table.scroll_view.home()
        if keycode[0] == 279:   # End
            print(keycode)
            self.my_table.scroll_view.end()


class TestApp(App):
    """ App class """
    def build(self):
        return MainScreen()

    def on_pause(self):
        return True


class TableApp(App):
    def build(self):
        # create a default grid layout with custom width/height
        table = TableView(size=(500,2320), pos_hint={'x':0.1, 'center_y':.5})
        # columns
        table.add_column(TableColumn("Col1", key="1", hint_text='0'))
        table.add_column(TableColumn("Col2", key="2", hint_text='0'))
        # table.add_column(TableColumn("Col2", key="3", hint_text='0'))
        # table.add_column(TableColumn("Col2", key="4", hint_text='0'))
        # table.add_column(TableColumn("Col2", key="5", hint_text='0'))
        # content
        for i in range(10):
            row = {'1': str(2 * i + 0), '2': str(2 * i + 1)}
            table.add_row(row)
        return table


class HBoxLayoutExample(App):
    def build(self):
        layout = BoxLayout(padding=10)
        colors = [red, green, blue, purple]

        button = Button(text='Hello from Kivy',
                        background_color=green)
        button.bind(on_press=self.on_press_button)
        layout.add_widget(button)
        for i in range(5):
            btn = Button(text="Button #%s" % (i + 1),
                         background_color=red
                         )

            layout.add_widget(btn)
        return layout

    def on_press_button(self, instance):
        print('Вы нажали на кнопку!')


class MainApp(App):
    def build(self):
        graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
                      x_ticks_major=25, y_ticks_major=1,
                      y_grid_label=True, x_grid_label=True, padding=5,
                      x_grid=True, y_grid=True, xmin=-0, xmax=100, ymin=-1, ymax=1)

        plot = MeshLinePlot(color=[1, 0, 0, 1])
        plot.points = [(x, np.sin(x / 10.)) for x in range(0, 101)]

        graph.add_plot(plot)

        self.operators = ["/", "*", "+", "-"]
        self.last_was_operator = None
        self.last_button = None
        main_layout = BoxLayout(orientation="vertical")
        self.solution = TextInput(
            multiline=False, readonly=True, halign="right", font_size=55
        )
        main_layout.add_widget(self.solution)
        buttons = [
            ["7", "8", "9", "/"],
            ["4", "5", "6", "*"],
            ["1", "2", "3", "-"],
            [".", "0", "C", "+"],
        ]
        for row in buttons:
            h_layout = BoxLayout()
            for label in row:
                button = Button(
                    text=label,
                    pos_hint={"center_x": 0.5, "center_y": 0.5},
                )
                button.bind(on_press=self.on_button_press)
                h_layout.add_widget(button)
            main_layout.add_widget(h_layout)

        equals_button = Button(
            text="=", pos_hint={"center_x": 0.5, "center_y": 0.5}
        )
        equals_button.bind(on_press=self.on_solution)
        main_layout.add_widget(equals_button)

        return main_layout

    def on_button_press(self, instance):
        current = self.solution.text
        button_text = instance.text

        if button_text == "C":
            # Очистка виджета с решением
            self.solution.text = ""
        else:
            if current and (
                    self.last_was_operator and button_text in self.operators):
                # Не добавляйте два оператора подряд, рядом друг с другом
                return
            elif current == "" and button_text in self.operators:
                # Первый символ не может быть оператором
                return
            else:
                new_text = current + button_text
                self.solution.text = new_text
        self.last_button = button_text
        self.last_was_operator = self.last_button in self.operators

    def on_solution(self, instance):
        text = self.solution.text
        if text:
            solution = str(eval(self.solution.text))
            self.solution.text = solution



class GT(BoxLayout):

    """Receives custom widget from corresponding <name>.kv file"""
    label_widget = ObjectProperty()
    graph_widget = ObjectProperty()

    def __init__(self, **kwargs):
        print('GT // __init__')
        super(GT, self).__init__(**kwargs)
        Window.clearcolor = (0.9, 0.93, 0.95, 1)

    def do_action(self):
        print('GT // do_action')
        self.label_widget.text = 'Graph was clicked.' # This works
        self.info = 'Important info!'


class framework_app(App):
    def build(self):
        print('framework_app // body')

        x = np.linspace(-np.pi, np.pi, 201)
        plt.plot(x, np.sin(x))
        plt.xlabel('Angle [rad]')
        plt.ylabel('sin(x)')
        plt.axis('tight')
        plt.show()

        return GT()


def main():
    # app = MainApp()
    # app.run()
    #
    # framework_app().run()

    TableApp().run()
    # TestApp().run()

    # table = Table()
    # table.cols = 2
    # table.add_button_row('123','456')
    # table.add_row([Button, {'text':'button2',
    #                         'color_widget': [0, 0, .5, 1],
    #                         'color_click': [0, 1, 0, 1]
    #                        }],
    #               [TextInput, {'text':'textinput2',
    #                            'color_click': [1, 0, .5, 1]
    #                           }])
    # table.choose_row(3)
    # table.del_row(5)
    # table.grid.color = [1, 0, 0, 1]
    # table.grid.cells[1][1].text = 'edited textinput text'
    # table.grid.cells[3][0].height = 100
    # table.label_panel.labels[1].text = 'New name'

if __name__ == '__main__':
    main()
