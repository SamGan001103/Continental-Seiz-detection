"""Event list panel: table with Accept / Reject / Jump buttons.

Row selection is exposed so the MainWindow can cycle events via
keyboard shortcuts and the view auto-centers on the selected event.
"""
from PyQt5 import QtCore, QtWidgets


HEADERS = ['#', 'start (s)', 'stop (s)', 'p', 'status', '']


class EventList(QtWidgets.QWidget):
    accepted = QtCore.pyqtSignal(int)
    rejected = QtCore.pyqtSignal(int)
    jumpRequested = QtCore.pyqtSignal(int)
    selectionChanged = QtCore.pyqtSignal(int)  # event_id (-1 = none)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        self._summary = QtWidgets.QLabel('No events')
        self._summary.setStyleSheet('color:#555; padding:2px;')
        v.addWidget(self._summary)

        self.table = QtWidgets.QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._emit_selection)
        v.addWidget(self.table)

    def set_events(self, events):
        self.table.setRowCount(0)
        for ev in events:
            self._append_row(ev)
        self._refresh_summary(events)

    def _refresh_summary(self, events):
        n = len(events)
        if not n:
            self._summary.setText('No events proposed at this threshold')
            return
        acc = sum(1 for e in events if e['status'] == 'accepted')
        rej = sum(1 for e in events if e['status'] == 'rejected')
        ed = sum(1 for e in events if e['status'] == 'edited')
        prop = n - acc - rej - ed
        self._summary.setText(
            '{} events — {} proposed · {} accepted · {} edited · {} rejected'
            .format(n, prop, acc, ed, rej))

    def _append_row(self, ev):
        row = self.table.rowCount()
        self.table.insertRow(row)

        def cell(txt):
            it = QtWidgets.QTableWidgetItem(str(txt))
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            return it

        self.table.setItem(row, 0, cell(ev['id']))
        self.table.setItem(row, 1, cell('{:.1f}'.format(ev['start'])))
        self.table.setItem(row, 2, cell('{:.1f}'.format(ev['stop'])))
        self.table.setItem(row, 3, cell('{:.2f}'.format(ev['prob'])))
        self.table.setItem(row, 4, cell(ev['status']))

        btns = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(btns)
        bl.setContentsMargins(2, 0, 2, 0)
        bl.setSpacing(2)
        jump = QtWidgets.QToolButton()
        jump.setText('→')
        jump.setToolTip('Jump (Enter)')
        accept = QtWidgets.QToolButton()
        accept.setText('✓')
        accept.setToolTip('Accept (Space)')
        reject = QtWidgets.QToolButton()
        reject.setText('✗')
        reject.setToolTip('Reject (X)')
        for b in (jump, accept, reject):
            b.setAutoRaise(True)
            b.setFixedWidth(22)
        jump.clicked.connect(lambda _=False, eid=ev['id']: self.jumpRequested.emit(eid))
        accept.clicked.connect(lambda _=False, eid=ev['id']: self.accepted.emit(eid))
        reject.clicked.connect(lambda _=False, eid=ev['id']: self.rejected.emit(eid))
        bl.addWidget(jump)
        bl.addWidget(accept)
        bl.addWidget(reject)
        self.table.setCellWidget(row, 5, btns)

    def update_row(self, event_id, ev):
        for row in range(self.table.rowCount()):
            if int(self.table.item(row, 0).text()) == event_id:
                self.table.item(row, 1).setText('{:.1f}'.format(ev['start']))
                self.table.item(row, 2).setText('{:.1f}'.format(ev['stop']))
                self.table.item(row, 4).setText(ev['status'])
                return

    # -----------------------------------------------------------------
    def selected_event_id(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return -1
        return int(self.table.item(rows[0].row(), 0).text())

    def _emit_selection(self):
        self.selectionChanged.emit(self.selected_event_id())

    def select_by_id(self, event_id):
        for row in range(self.table.rowCount()):
            if int(self.table.item(row, 0).text()) == event_id:
                self.table.selectRow(row)
                return

    def cycle_selection(self, delta):
        """delta = +1 next, -1 prev."""
        n = self.table.rowCount()
        if not n:
            return -1
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            target = 0 if delta > 0 else n - 1
        else:
            target = (rows[0].row() + delta) % n
        self.table.selectRow(target)
        return int(self.table.item(target, 0).text())
